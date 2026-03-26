import gc
import json
import os
import subprocess

from google.colab import auth

auth.authenticate_user()

os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/content/hf_datasets_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/transformers_cache"

import gcsfs
from datasets import Dataset, load_dataset
from tqdm import tqdm

GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/vqa-vs-preprocessed"
SHARD_SIZE = 256

OOD_CATEGORIES = ["KO", "KOP", "KW+KO", "KW", "KWP", "QT+KO", "QT+KW+KO", "QT+KW", "QT"]

OOD_QUESTION_FILES = [
    f"gs://{GCS_BUCKET}/datasets/vqa-vs/OOD-Test/{cat}/OOD-Test-{cat}-Ques.json"
    for cat in OOD_CATEGORIES
]
OOD_ANSWER_FILES = [p.replace("-Ques.json", "-Ans.json") for p in OOD_QUESTION_FILES]

fs = gcsfs.GCSFileSystem(token="google_default")


def format_eval_sample(example: dict) -> dict:
    """Return a raw eval sample retaining the PIL image and all answer annotations."""
    return {
        "image": example["image"],
        "question": example["question"],
        "answers": example["answers"],
        "multiple_choice_answer": example.get("multiple_choice_answer", ""),
        "question_id": example.get("question_id"),
        "image_id": example.get("image_id"),
        "question_type": example.get("question_type", ""),
        "answer_type": example.get("answer_type", ""),
    }


def download_json_from_gcs(gcs_path: str) -> dict | list:
    """Read and parse a JSON file from GCS."""
    with fs.open(gcs_path, "r") as f:
        return json.load(f)


def print_disk_usage() -> None:
    """Print current overlay filesystem disk usage."""
    result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "overlay" in line:
            print(f"Disk: {line}")


print("=" * 70)
print("STEP 1: Loading all OOD questions to find needed image IDs")
print("=" * 70)

all_img_id_to_questions: dict[int, list[dict]] = {}

for ques_path, ans_path in zip(OOD_QUESTION_FILES, OOD_ANSWER_FILES):
    category = ques_path.split("/")[-2]
    print(f"\nLoading {category}...")

    questions = download_json_from_gcs(ques_path)
    answers = download_json_from_gcs(ans_path)
    answer_dict = {a["question_id"]: a for a in answers}

    for q in questions:
        img_id = q["image_id"]
        ans_entry = answer_dict.get(q["question_id"])
        if not ans_entry:
            continue
        all_img_id_to_questions.setdefault(img_id, []).append({
            "category": category,
            "question": q,
            "answer_entry": ans_entry,
        })

    print(f"Loaded {len(questions)} questions")

needed_img_ids = set(all_img_id_to_questions.keys())
print(f"\nTotal unique images needed across all OOD sets: {len(needed_img_ids)}")

print("\n" + "=" * 70)
print("STEP 2: Streaming VQAv2 validation to collect images")
print("=" * 70)

vqav2_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)

category_samples: dict[str, list[dict]] = {cat: [] for cat in OOD_CATEGORIES}
found_img_ids: set[int] = set()

for vqa_sample in tqdm(vqav2_stream, desc="Matching images", total=214354):
    img_id = vqa_sample["image_id"]
    if isinstance(img_id, str) and "_" in img_id:
        img_id = int(img_id.split("_")[-1])

    if img_id in needed_img_ids and img_id not in found_img_ids:
        found_img_ids.add(img_id)

        for item in all_img_id_to_questions[img_id]:
            ans_entry = item["answer_entry"]
            category_samples[item["category"]].append({
                "image": vqa_sample["image"],
                "image_id": img_id,
                "question": item["question"]["question"],
                "question_id": item["question"]["question_id"],
                "answers": ans_entry.get("answers", []),
                "multiple_choice_answer": ans_entry.get("multiple_choice_answer", ""),
                "question_type": item["question"].get("question_type", ""),
                "answer_type": ans_entry.get("answer_type", ""),
            })

    if len(found_img_ids) == len(needed_img_ids):
        print(f"\nFound all {len(needed_img_ids)} images")
        break

print("Collected samples for all categories")
for cat in OOD_CATEGORIES:
    print(f"  {cat}: {len(category_samples[cat])} samples")

del vqav2_stream, all_img_id_to_questions
gc.collect()

print("\n" + "=" * 70)
print("STEP 3: Processing and saving each category to shards")
print("=" * 70)

for category in OOD_CATEGORIES:
    samples = category_samples[category]

    if not samples:
        print(f"\n{category}: no samples, skipping")
        continue

    print(f"\n{'='*70}")
    print(f"Processing {category}: {len(samples)} samples")
    print(f"{'='*70}")

    first = samples[0]
    print(f"Answer count: {len(first['answers'])}, first: {first['answers'][0]}, MC: {first['multiple_choice_answer']}")

    split_name = f"ood_{category.lower().replace('+', '_')}"
    gcs_split_dir = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{split_name}"
    fs_split_prefix = f"{GCS_BUCKET}/{GCS_PREFIX}/{split_name}"

    existing = sorted(fs.glob(f"{fs_split_prefix}/*.parquet"))
    if existing:
        print(f"Deleting {len(existing)} old shards")
        for path in existing:
            fs.rm(path)

    num_shards = (len(samples) + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"Processing {len(samples)} samples into {num_shards} shards...")

    for shard_idx in range(num_shards):
        start_idx = shard_idx * SHARD_SIZE
        end_idx = min(start_idx + SHARD_SIZE, len(samples))
        chunk = samples[start_idx:end_idx]

        print(f"\n  Shard {shard_idx}/{num_shards - 1}: samples {start_idx + 1}-{end_idx}")

        formatted_chunk = []
        for sample in tqdm(chunk, desc="Formatting"):
            try:
                formatted_chunk.append(format_eval_sample(sample))
            except Exception as e:
                print(f"Skipped sample: {e}")

        if not formatted_chunk:
            continue

        out_path = f"{gcs_split_dir}/shard_{shard_idx:04d}.parquet"
        print(f"  Saving to {out_path}")
        Dataset.from_list(formatted_chunk).to_parquet(out_path)

        del formatted_chunk
        gc.collect()

    print(f"{split_name} complete")

    del samples
    category_samples[category] = []
    gc.collect()
    print_disk_usage()

print("\n" + "=" * 70)
print("ALL OOD CATEGORIES PROCESSED")
print("=" * 70)
print(f"\nPreprocessed datasets saved to: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
print("\nDatasets available:")
for cat in OOD_CATEGORIES:
    print(f"  ood_{cat.lower().replace('+', '_')}")