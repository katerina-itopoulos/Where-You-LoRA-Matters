import base64
import io
import json
import os
from collections import Counter
from pathlib import Path

import gcsfs
import pandas as pd
from datasets import load_dataset
from google.cloud import storage
from PIL import Image

BUCKET_NAME = "where_you_lora_matters_thesis"

CONFIG_FOLDER_MAP_SEED42: dict[str, str] = {
    "baseline":         "baseline_seed42",
    "llm_only":         "llm_seed_42",
    "projector_only":   "proj_1e-4_seed_42",
    "vision_projector": "vision_proj_seed42",
    "llm_proj":         "llm_proj_seed_42",
    "llm_vision":       "llm_vision_seed_42",
    "llm_proj_vision":  "llm_proj_vision_seed42",
}

CONFIG_FOLDER_MAP_SEED123: dict[str, str] = {
    "baseline":         "baseline_seed123",
    "llm_only":         "llm_seed123",
    "projector_only":   "proj_1e-4_seed_123",
    "vision_projector": "vision_proj_seed_123",
    "llm_proj":         "llm_proj_seed_123",
    "llm_vision":       "llm_vision_seed_123",
    "llm_proj_vision":  "llm_proj_vision_seed_123",
}

SEED_CONFIGS: dict[str, dict[str, str]] = {
    "seed42":  CONFIG_FOLDER_MAP_SEED42,
    "seed123": CONFIG_FOLDER_MAP_SEED123,
}

BENCHMARK_FILE_MAP: dict[str, str] = {
    "dash_b":          "dash_b_results.json",
    "hallusionbench":  "hallusionbench_image_results.json",
    "mmvp":            "mmvp_results.json",
    "vqa_vs_ko":       "vqa_vs_ood_ko_results.json",
    "vqa_vs_kop":      "vqa_vs_ood_kop_results.json",
    "vqa_vs_kw":       "vqa_vs_ood_kw_results.json",
    "vqa_vs_kwp":      "vqa_vs_ood_kwp_results.json",
    "vqa_vs_qt":       "vqa_vs_ood_qt_results.json",
    "vqa_vs_kw_ko":    "vqa_vs_ood_kw_ko_results.json",
    "vqa_vs_qt_ko":    "vqa_vs_ood_qt_ko_results.json",
    "vqa_vs_qt_kw":    "vqa_vs_ood_qt_kw_results.json",
    "vqa_vs_qt_kw_ko": "vqa_vs_ood_qt_kw_ko_results.json",
    "vqav2":           "vqav2_test_results.json",
}

PROJECTOR_CONFIGS: set[str] = {"projector_only", "vision_projector"}
LLM_CONFIGS: set[str] = {"llm_only", "llm_proj", "llm_vision", "llm_proj_vision"}

VQA_VS_PARQUET_MAP: dict[str, str] = {
    "vqa_vs_ko":       f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_ko/*.parquet",
    "vqa_vs_kop":      f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_kop/*.parquet",
    "vqa_vs_kw":       f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_kw/*.parquet",
    "vqa_vs_kwp":      f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_kwp/*.parquet",
    "vqa_vs_qt":       f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_qt/*.parquet",
    "vqa_vs_kw_ko":    f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_kw_ko/*.parquet",
    "vqa_vs_qt_ko":    f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_qt_ko/*.parquet",
    "vqa_vs_qt_kw":    f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_qt_kw/*.parquet",
    "vqa_vs_qt_kw_ko": f"{BUCKET_NAME}/datasets/vqa-vs-preprocessed/ood_qt_kw_ko/*.parquet",
}

HF_DATASET_CONFIG: dict[str, dict[str, str]] = {
    "hallusionbench": {"path": "lmms-lab/HallusionBench", "split": "image"},
    "dash_b":         {"path": "YanNeu/DASH-B",           "split": "test"},
}

VQAV2_TRAIN_SIZE = 20000
VQAV2_VAL_SIZE = 2000
VQAV2_TEST_SIZE = 2000

fs = gcsfs.GCSFileSystem()
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


def load_predictions(bucket, config_folder: str, benchmark_file: str) -> list[dict] | None:
    """Download and parse predictions from GCS, assigning sequential question IDs where needed."""
    path = f"experiments_results_final/{config_folder}/results/{benchmark_file}"
    blob = bucket.blob(path)
    if not blob.exists():
        return None

    data = json.loads(blob.download_as_bytes().decode("utf-8"))
    preds = data if isinstance(data, list) else data.get("predictions", [])

    for i, p in enumerate(preds):
        if "hallusionbench" in benchmark_file or "question_id" not in p:
            p["question_id"] = i

    return preds


def load_all_predictions(
    bucket,
    benchmark_key: str,
    config_folder_map: dict[str, str],
) -> dict[str, dict[int, dict]]:
    """Return a dict mapping config name to a question-ID-keyed prediction dict."""
    bench_file = BENCHMARK_FILE_MAP[benchmark_key]
    all_preds = {}
    for config_name, folder in config_folder_map.items():
        preds = load_predictions(bucket, folder, bench_file)
        if preds is None:
            print(f"  {config_name} missing for {benchmark_key}")
            continue
        all_preds[config_name] = {p["question_id"]: p for p in preds}
    return all_preds


def is_correct(prediction: str, reference: list | str) -> bool:
    """Return True if the prediction matches the reference under VQA or exact-match scoring."""
    if isinstance(reference, list):
        answers = [r["answer"].strip().lower() for r in reference]
        return answers.count(prediction.strip().lower()) >= 3
    return prediction.strip().lower() == str(reference).strip().lower()


def find_all_cases(all_preds: dict[str, dict[int, dict]]) -> dict[str, list[int]]:
    """Partition question IDs into analysis categories based on correctness patterns."""
    all_qids = set.intersection(*[set(p.keys()) for p in all_preds.values()])

    projector_fails_llm_succeeds, all_agree_correct, all_disagree, other = [], [], [], []
    proj_configs = [c for c in PROJECTOR_CONFIGS if c in all_preds]
    llm_configs = [c for c in LLM_CONFIGS if c in all_preds]

    for qid in sorted(all_qids):
        ref = list(all_preds.values())[0][qid]["reference"]
        proj_correct = [is_correct(all_preds[c][qid]["prediction"], ref) for c in proj_configs]
        llm_correct = [is_correct(all_preds[c][qid]["prediction"], ref) for c in llm_configs]
        all_correct = [is_correct(all_preds[c][qid]["prediction"], ref) for c in all_preds]

        if not (proj_configs and llm_configs):
            continue
        if not any(proj_correct) and all(llm_correct):
            projector_fails_llm_succeeds.append(qid)
        elif all(all_correct):
            all_agree_correct.append(qid)
        elif len({all_preds[c][qid]["prediction"] for c in all_preds}) > 2:
            all_disagree.append(qid)
        else:
            other.append(qid)

    return {
        "projector_fails_llm_succeeds": projector_fails_llm_succeeds,
        "all_agree_correct": all_agree_correct,
        "all_disagree": all_disagree,
        "other": other,
    }


def dict_to_pil(img_dict: dict) -> Image.Image:
    """Convert a bytes-dict image representation to a PIL Image."""
    return Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")


def load_mmvp_images(needed_qids: set[int]) -> dict[int, Image.Image]:
    """Load MMVP images from HuggingFace using filename-sorted index matching."""
    print(f"Loading MMVP images from HF (filename-sorted)...")
    img_ds = load_dataset("MMVP/MMVP")["train"]

    photo_id_to_img = {}
    for i in range(len(img_ds)):
        img = img_ds[i]["image"]
        photo_id = int(os.path.basename(img.filename).replace(".jpg", ""))
        photo_id_to_img[photo_id] = img

    images = {qid: photo_id_to_img[qid + 1].convert("RGB") for qid in needed_qids if qid + 1 in photo_id_to_img}
    print(f"Loaded {len(images)}/{len(needed_qids)} MMVP images")
    return images


def load_vqav2_images(needed_qids: set[int]) -> dict[int, Image.Image]:
    """Stream VQAv2 validation, skip train and val splits, match by question ID."""
    print(f"Loading VQAv2 test images (streaming, skip {VQAV2_TRAIN_SIZE + VQAV2_VAL_SIZE})...")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    ds = ds.skip(VQAV2_TRAIN_SIZE + VQAV2_VAL_SIZE)

    needed = set(needed_qids)
    images = {}
    for idx, example in enumerate(ds):
        if idx >= VQAV2_TEST_SIZE:
            break
        qid = example["question_id"]
        if qid in needed:
            img = example["image"]
            images[qid] = img.convert("RGB") if hasattr(img, "convert") else img
        if len(images) == len(needed):
            break

    print(f"Loaded {len(images)}/{len(needed)} VQAv2 images")
    return images


def load_vqa_vs_images(benchmark_key: str, needed_qids: set[int]) -> dict[int, Image.Image]:
    """Load VQA-VS images from preprocessed GCS parquet shards."""
    glob_path = VQA_VS_PARQUET_MAP.get(benchmark_key)
    if not glob_path:
        print(f"No parquet path for {benchmark_key}")
        return {}

    shards = sorted(fs.glob(glob_path))
    if not shards:
        print(f"No shards found at {glob_path}")
        return {}

    print(f"Loading {benchmark_key} from {len(shards)} parquet shards...")
    images = {}
    for shard_path in shards:
        df = pd.read_parquet(f"gs://{shard_path}", filesystem=fs)
        for _, row in df.iterrows():
            qid = row["question_id"]
            if qid in needed_qids:
                images[qid] = dict_to_pil(row["image"])
        if set(needed_qids).issubset(images.keys()):
            break

    print(f"Loaded {len(images)}/{len(needed_qids)} images")
    return images


def load_hf_images(benchmark_key: str, needed_qids: set[int]) -> dict[int, Image.Image]:
    """Load images from HuggingFace for HallusionBench and DASH-B by sequential index."""
    cfg = HF_DATASET_CONFIG[benchmark_key]
    print(f"Loading {benchmark_key} from HF ({cfg['path']})...")
    ds = load_dataset(cfg["path"])[cfg["split"]]

    img_col = "image" if "image" in ds.features else next(
        (c for c in ds.features if "image" in c.lower()), None
    )
    if img_col is None:
        print("No image column found")
        return {}

    print(f"Image column: '{img_col}', {len(ds)} total samples")
    images = {}
    for qid in needed_qids:
        if qid >= len(ds):
            continue
        img = ds[qid][img_col]
        if hasattr(img, "convert"):
            images[qid] = img.convert("RGB")
        elif isinstance(img, dict) and img.get("bytes"):
            images[qid] = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        else:
            print(f"Unknown image format for qid {qid}: {type(img)}")

    print(f"Loaded {len(images)}/{len(needed_qids)} images")
    return images


def load_images_for_benchmark(benchmark_key: str, needed_qids: set[int]) -> dict[int, Image.Image]:
    """Dispatch to the appropriate image loader for the given benchmark."""
    if benchmark_key == "mmvp":
        return load_mmvp_images(needed_qids)
    if benchmark_key == "vqav2":
        return load_vqav2_images(needed_qids)
    if benchmark_key in VQA_VS_PARQUET_MAP:
        return load_vqa_vs_images(benchmark_key, needed_qids)
    if benchmark_key in HF_DATASET_CONFIG:
        return load_hf_images(benchmark_key, needed_qids)
    print(f"No image loader for {benchmark_key}")
    return {}


def img_to_base64(pil_img: Image.Image, max_size: int = 250) -> str:
    """Thumbnail and encode a PIL image as a base64 JPEG string."""
    pil_img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def render_section(
    title: str,
    qids: list[int],
    color: str,
    all_preds: dict[str, dict[int, dict]],
    images: dict[int, Image.Image],
) -> str:
    """Return an HTML section showing predictions and images for a list of question IDs."""
    if not qids:
        return f'<h2 style="color:{color}">{title}</h2><p><em>No examples found.</em></p>'

    config_names = list(all_preds.keys())

    def format_reference(ref: list | str) -> str:
        if isinstance(ref, list):
            counts = Counter(r["answer"] for r in ref)
            return ", ".join(f"{a} ({n})" for a, n in counts.most_common(3))
        return str(ref)

    def correct_style(pred: str, ref: list | str) -> str:
        correct = is_correct(pred, ref)
        bg = "#d4edda" if correct else "#f8d7da"
        return f'style="background:{bg}; padding:3px 6px; border-radius:3px; font-weight:bold;"'

    rows = []
    for qid in qids:
        first_pred = list(all_preds.values())[0][qid]
        ref = first_pred["reference"]
        question = first_pred["question"].replace(" Answer with yes or no only.", "").strip()

        img_cell = (
            f'<img src="data:image/jpeg;base64,{img_to_base64(images[qid])}" '
            f'style="max-width:180px; max-height:180px;">'
            if qid in images
            else f'<span style="color:gray; font-size:11px;">No image<br/>(ID: {qid})</span>'
        )

        pred_cells = "".join(
            f'<td {correct_style(all_preds[cfg][qid]["prediction"], ref)}>'
            f'{all_preds[cfg][qid]["prediction"]}</td>'
            if qid in all_preds.get(cfg, {})
            else '<td style="color:gray">—</td>'
            for cfg in config_names
        )

        rows.append(f"""
        <tr>
            <td style="text-align:center;">{img_cell}</td>
            <td style="max-width:220px; font-size:11px; text-align:left;">{question}</td>
            <td style="font-weight:bold; text-align:center; font-size:11px;">{format_reference(ref)}</td>
            {pred_cells}
        </tr>
        """)

    header_cells = "".join(
        f'<th style="background:#495057; color:white; padding:6px;">{c}</th>'
        for c in config_names
    )

    return f"""
    <h2 style="color:{color}; margin-top:50px; border-bottom:2px solid {color}; padding-bottom:6px;">
        {title}
        <span style="font-size:13px; color:gray; font-weight:normal;">— {len(qids)} examples</span>
    </h2>
    <table border="1" cellpadding="6" cellspacing="0"
           style="border-collapse:collapse; width:100%; font-size:12px; margin-top:12px;">
        <thead>
            <tr style="background:#343a40; color:white;">
                <th>Image</th><th>Question</th><th>Reference</th>
                {header_cells}
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """


def generate_html_report(
    benchmark_key: str,
    seed_label: str,
    all_preds: dict[str, dict[int, dict]],
    cases: dict[str, list[int]],
    images: dict[int, Image.Image],
    output_dir: str = "qualitative_reports",
) -> Path:
    """Write an HTML qualitative report locally and upload it to GCS."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sections = "".join([
        render_section("Projector Configs Fail — LLM Configs Succeed",
                       cases["projector_fails_llm_succeeds"], "#dc3545", all_preds, images),
        render_section("All Configs Agree and Correct",
                       cases["all_agree_correct"], "#28a745", all_preds, images),
        render_section("Configs Disagree",
                       cases["all_disagree"], "#ffc107", all_preds, images),
        render_section("Other",
                       cases.get("other", []), "#6c757d", all_preds, images),
    ])

    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Qualitative Analysis — {benchmark_key} ({seed_label})</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; max-width: 1400px; }}
            td, th {{ vertical-align: middle; text-align: center; }}
            h1 {{ color: #212529; border-bottom: 3px solid #212529; padding-bottom: 10px; }}
            p.legend {{ color: gray; font-size: 13px; }}
        </style>
    </head>
    <body>
        <h1>Qualitative Analysis: {benchmark_key.upper()} — {seed_label}</h1>
        <p class="legend">Green = correct &nbsp;|&nbsp; Red = incorrect &nbsp;|&nbsp; {seed_label}</p>
        {sections}
    </body>
    </html>"""

    out_path = Path(output_dir) / f"{benchmark_key}_qualitative_{seed_label}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")

    gcs_path = f"qualitative_reports/{benchmark_key}_qualitative_{seed_label}.html"
    bucket.blob(gcs_path).upload_from_string(html, content_type="text/html")
    print(f"Uploaded to gs://{BUCKET_NAME}/{gcs_path}")

    return out_path


for benchmark_key in BENCHMARK_FILE_MAP:
    print(f"\n{'='*55}")
    print(f"Processing: {benchmark_key}")

    all_preds_per_seed: dict[str, dict[str, dict[int, dict]]] = {}
    cases_per_seed: dict[str, dict[str, list[int]]] = {}

    for seed_label, config_folder_map in SEED_CONFIGS.items():
        print(f"\n  Seed: {seed_label}")
        all_preds = load_all_predictions(bucket, benchmark_key, config_folder_map)
        if not all_preds:
            print(f"  No predictions found for {seed_label}, skipping")
            continue
        all_preds_per_seed[seed_label] = all_preds
        cases_per_seed[seed_label] = find_all_cases(all_preds)

    if not all_preds_per_seed:
        continue

    needed_qids: set[int] = set()
    for cases in cases_per_seed.values():
        for qid_list in cases.values():
            needed_qids.update(qid_list)

    print(f"\n  Loading images for {len(needed_qids)} unique questions (shared across seeds)...")
    images = load_images_for_benchmark(benchmark_key, needed_qids)

    for seed_label, all_preds in all_preds_per_seed.items():
        cases = cases_per_seed[seed_label]
        print(f"\n  Generating report for {seed_label}...")
        for category, qids in cases.items():
            print(f"    {category}: {len(qids)}")
        generate_html_report(
            benchmark_key=benchmark_key,
            seed_label=seed_label,
            all_preds=all_preds,
            cases=cases,
            images=images,
            output_dir="qualitative_reports",
        )

print("\nAll reports saved to qualitative_reports/")