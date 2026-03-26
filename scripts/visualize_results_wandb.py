import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gcsfs
import numpy as np
import torch
import wandb

from src.collapse_metrics import summarize_vectors

GCS_BUCKET = "where_you_lora_matters_thesis"

EXPERIMENTS: dict[str, list[str]] = {
    "baseline":         ["experiments_results_final/baseline_seed42",        "experiments_results_final/baseline_seed123"],
    "llm_only":         ["experiments_results_final/llm_seed_42",            "experiments_results_final/llm_seed123"],
    "llm_proj":         ["experiments_results_final/llm_proj_seed_42",       "experiments_results_final/llm_proj_seed_123"],
    "projector_only":   ["experiments_results_final/proj_1e-4_seed_42",      "experiments_results_final/proj_1e-4_seed_123"],
    "vision_projector": ["experiments_results_final/vision_proj_seed42",     "experiments_results_final/vision_proj_seed_123"],
    "vision_llm":       ["experiments_results_final/llm_vision_seed_42",     "experiments_results_final/llm_vision_seed_123"],
    "llm_proj_vision":  ["experiments_results_final/llm_proj_vision_seed42", "experiments_results_final/llm_proj_vision_seed_123"],
}

STABILITY_METRICS: list[str] = [
    "accuracy", "f1",
    "cka_img_txt",
    "norm_erank_img", "norm_erank_txt",
    "cr1_img", "cr1_txt",
    "modality_gap_mean",
]

WANDB_PROJECT = "where-you-lora-matters"
WANDB_ENTITY: str | None = None
WANDB_RUN_NAME = "results_3000steps_avg_seeds_final"

fs = gcsfs.GCSFileSystem(token="google_default")


def load_features_from_gcs(feature_file: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Download a feature .npz from GCS and return Vproj and T_fused tensors."""
    local_path = f"/tmp/{feature_file.split('/')[-1]}"
    fs.get(feature_file, local_path)
    data = np.load(local_path, allow_pickle=True)
    Vproj = torch.from_numpy(np.asarray(data["Vproj"], dtype=np.float32))
    T_fused = torch.from_numpy(np.asarray(data["T_fused"], dtype=np.float32))
    data.close()
    return Vproj, T_fused


def load_json_from_gcs(json_file: str) -> dict:
    """Read and parse a JSON file from GCS."""
    with fs.open(json_file, "r") as f:
        return json.load(f)


def fmt(value: object) -> str:
    """Format a scalar value for display, returning 'nan' for missing or NaN values."""
    if value is None or (isinstance(value, float) and value != value):
        return "nan"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def pool_two_seeds(row1: dict, row2: dict) -> dict:
    """
    Merge metrics from two seeds into a single row.

    Mean/std pairs are combined via pooled mean and pooled std.
    Scalar metrics are averaged. The ``model`` key is taken from ``row1``.
    """
    all_keys = set(row1.keys()) | set(row2.keys())
    merged = {"model": row1.get("model", row2.get("model"))}
    processed = {"model"}

    for key in sorted(all_keys):
        if key in processed:
            continue

        if key.endswith("_mean"):
            base = key[:-5]
            s_key = base + "_std"
            m1, m2 = row1.get(key, float("nan")), row2.get(key, float("nan"))
            s1, s2 = row1.get(s_key, float("nan")), row2.get(s_key, float("nan"))
            merged[key] = (m1 + m2) / 2 if not (np.isnan(m1) or np.isnan(m2)) else float("nan")
            merged[s_key] = (
                np.sqrt((s1**2 + s2**2) / 2)
                if not (np.isnan(s1) or np.isnan(s2))
                else float("nan")
            )
            processed.update([key, s_key])

        elif key.endswith("_std"):
            v1, v2 = row1.get(key, float("nan")), row2.get(key, float("nan"))
            merged[key] = (v1 + v2) / 2 if not (np.isnan(v1) or np.isnan(v2)) else float("nan")
            processed.add(key)

        else:
            v1, v2 = row1.get(key, float("nan")), row2.get(key, float("nan"))
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                merged[key] = (v1 + v2) / 2 if not (np.isnan(v1) or np.isnan(v2)) else float("nan")
            else:
                merged[key] = v1
            processed.add(key)

    return merged


def stability_row(row1: dict, row2: dict, metrics: list[str]) -> dict:
    """Return a dict of ``'mean ± std'`` strings for the given metrics across two seeds."""
    result = {"model": row1.get("model", row2.get("model"))}
    for metric in metrics:
        v1, v2 = row1.get(metric, float("nan")), row2.get(metric, float("nan"))
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and not np.isnan(v1) and not np.isnan(v2):
            result[metric] = f"{(v1 + v2) / 2:.3f} ± {abs(v1 - v2) / 2:.3f}"
        else:
            result[metric] = "nan"
    return result


def build_wandb_table(rows: list[dict]) -> wandb.Table:
    """Format a list of metric dicts into a W&B Table, collapsing mean/std pairs into one column."""
    all_columns = sorted({k for row in rows for k in row.keys()})

    formatted_columns = ["model"]
    for col in all_columns:
        if col == "model":
            continue
        if col.endswith("_mean"):
            base = col[:-5]
            formatted_columns.append(base if base + "_std" in all_columns else col)
        elif not col.endswith("_std"):
            formatted_columns.append(col)
    formatted_columns = list(dict.fromkeys(formatted_columns))

    data = []
    for row in rows:
        formatted_row = []
        for col in formatted_columns:
            if col + "_mean" in row and col + "_std" in row:
                m, s = row[col + "_mean"], row[col + "_std"]
                formatted_row.append(f"{m:.3f} ± {s:.3f}" if not (np.isnan(m) or np.isnan(s)) else "nan")
            elif col in row:
                formatted_row.append(fmt(row[col]))
            else:
                formatted_row.append("nan")
        data.append(formatted_row)

    return wandb.Table(columns=formatted_columns, data=data)


print("=" * 80)
print("META EVALUATION: ALL LORA CONFIGS - AVERAGED ACROSS SEEDS")
print("=" * 80)

wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    config={"experiments": list(EXPERIMENTS.keys())},
)

all_tables_by_seed: dict[str, dict[str, list[dict]]] = {}

for model_name, seed_dirs in EXPERIMENTS.items():
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"{'='*80}")

    for seed_idx, output_dir in enumerate(seed_dirs):
        seed_label = "seed42" if seed_idx == 0 else "seed123"
        print(f"\n  {seed_label}: {output_dir}")

        eval_summary_path = f"{GCS_BUCKET}/{output_dir}/evaluation_summary.json"
        try:
            eval_summary = load_json_from_gcs(eval_summary_path)
            all_task_results = eval_summary.get("results", {})
            print(f"  Loaded summary with {len(all_task_results)} datasets")
        except Exception as e:
            print(f"  Could not load summary: {e}")
            all_task_results = {}

        features_path = f"{GCS_BUCKET}/{output_dir}/features/"
        try:
            feature_files = sorted(f for f in fs.ls(features_path) if f.endswith(".npz"))
            print(f"  Found {len(feature_files)} feature files")
        except Exception as e:
            print(f"  Could not list features: {e}")
            feature_files = []

        for feature_file in feature_files:
            dataset_name = feature_file.split("/")[-1].replace("_features.npz", "")
            print(f"\n  Dataset: {dataset_name}")

            try:
                Vproj, T_fused = load_features_from_gcs(feature_file)
                collapse_metrics = summarize_vectors(Vproj, T_fused)
                print("  Collapse metrics computed")
            except Exception as e:
                print(f"  Could not load features: {e}")
                collapse_metrics = {}

            row = {
                "model": model_name,
                "seed": seed_label,
                **all_task_results.get(dataset_name, {}),
                **collapse_metrics,
            }

            all_tables_by_seed.setdefault(dataset_name, {}).setdefault(model_name, []).append(row)


all_tables: dict[str, list[dict]] = {}
all_tables_seed42: dict[str, list[dict]] = {}
all_tables_seed123: dict[str, list[dict]] = {}
all_stability: dict[str, list[dict]] = {}

for dataset_name, model_rows in all_tables_by_seed.items():
    all_tables[dataset_name] = []
    all_tables_seed42[dataset_name] = []
    all_tables_seed123[dataset_name] = []
    all_stability[dataset_name] = []

    for model_name, seed_rows in model_rows.items():
        if len(seed_rows) == 2:
            merged = pool_two_seeds(seed_rows[0], seed_rows[1])
            stab = stability_row(seed_rows[0], seed_rows[1], STABILITY_METRICS)
            print(f"  {model_name} / {dataset_name}: merged 2 seeds")
        elif len(seed_rows) == 1:
            merged = seed_rows[0]
            stab = {"model": model_name, **{m: "only 1 seed" for m in STABILITY_METRICS}}
            print(f"  {model_name} / {dataset_name}: only 1 seed available")
        else:
            print(f"  {model_name} / {dataset_name}: no data")
            continue

        all_tables[dataset_name].append(merged)
        all_stability[dataset_name].append(stab)

        for row in seed_rows:
            row_clean = {k: v for k, v in row.items() if k != "seed"}
            if row.get("seed") == "seed42":
                all_tables_seed42[dataset_name].append(row_clean)
            else:
                all_tables_seed123[dataset_name].append(row_clean)


for dataset_name in all_tables:
    print(f"\n{'='*80}")
    print(f"Logging tables for: {dataset_name}")

    if all_tables[dataset_name]:
        wandb.log({f"table_{dataset_name}": build_wandb_table(all_tables[dataset_name])})
        print("  Logged averaged table")

    if all_tables_seed42[dataset_name]:
        wandb.log({f"table_{dataset_name}_seed42": build_wandb_table(all_tables_seed42[dataset_name])})
        print("  Logged seed42 table")

    if all_tables_seed123[dataset_name]:
        wandb.log({f"table_{dataset_name}_seed123": build_wandb_table(all_tables_seed123[dataset_name])})
        print("  Logged seed123 table")

    if all_stability[dataset_name]:
        stab_columns = ["model"] + STABILITY_METRICS
        stab_data = [[row.get(col, "nan") for col in stab_columns] for row in all_stability[dataset_name]]
        wandb.log({f"table_{dataset_name}_stability": wandb.Table(columns=stab_columns, data=stab_data)})
        print("  Logged stability table")

wandb.finish()

print("\n" + "=" * 80)
print("META ANALYSIS COMPLETE")
print("=" * 80)