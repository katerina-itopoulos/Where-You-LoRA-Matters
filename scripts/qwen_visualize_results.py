import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gcsfs
import torch
import numpy as np
import json
import wandb

from src.collapse_metrics import summarize_vectors

# ================================
# CONFIG
# ================================

GCS_BUCKET = "where_you_lora_matters_thesis"

# All experiments you want to compare
EXPERIMENTS = {
    "baseline": "experiments_qwen3vl/baseline",
    "llm_only": "experiments_qwen3vl/llm_only",
    "llm_proj": "experiments_qwen3vl/llm_proj",
    "projector_only": "experiments_qwen3vl/proj_only",
    "vision_projector": "experiments_qwen3vl/vision_proj",
    "vision_llm": "experiments_qwen3vl/vision_llm",
}

WANDB_PROJECT = "where-you-lora-matters"
WANDB_ENTITY = None
WANDB_RUN_NAME = "meta_eval_all_lora_configs"

fs = gcsfs.GCSFileSystem(token='google_default')

# ================================
# HELPERS
# ================================

def load_features_from_gcs(feature_file):
    """Robust loader: handles dtype=object npz files"""
    local_path = f"/tmp/{feature_file.split('/')[-1]}"
    fs.get(feature_file, local_path)
    
    data = np.load(local_path, allow_pickle=True)
    
    # Force proper numeric arrays
    Vproj = np.asarray(data["Vproj"], dtype=np.float32)
    T_fused = np.asarray(data["T_fused"], dtype=np.float32)
    
    data.close()
    
    return torch.from_numpy(Vproj), torch.from_numpy(T_fused)

def load_results_from_gcs(result_file):
    with fs.open(result_file, 'r') as f:
        return json.load(f)

def fmt(mean, std):
    if mean != mean or std != std:
        return "nan"
    return f"{mean:.3f} ± {std:.3f}"

# ================================
# MAIN
# ================================

print("="*80)
print("META EVALUATION: ALL LORA CONFIGS")
print("="*80)

wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    config={"experiments": list(EXPERIMENTS.keys())}
)

# Will store: { dataset_name: [ {model: ..., metric1: ..., ...}, ... ] }
all_tables = {}

for model_name, output_dir in EXPERIMENTS.items():
    print(f"\n{'='*80}")
    print(f"🔍 Processing model: {model_name}")
    print(f"{'='*80}")
    
    features_path = f"{GCS_BUCKET}/{output_dir}/features/"
    results_path = f"{GCS_BUCKET}/{output_dir}/results/"
    
    feature_files = sorted([f for f in fs.ls(features_path) if f.endswith('.npz')])
    
    for feature_file in feature_files:
        dataset_name = feature_file.split('/')[-1].replace('_features.npz', '')
        print(f"  → Dataset: {dataset_name}")
        
        # -------- Features / collapse metrics --------
        Vproj, T_fused = load_features_from_gcs(feature_file)
        collapse_metrics = summarize_vectors(Vproj, T_fused)
        
        # -------- Task metrics (FIXED - no predictions) --------
        result_file = f"{results_path}{dataset_name}_results.json"
        try:
            results = load_results_from_gcs(result_file)
            
            # Only get metrics, NOT predictions
            task_metrics = results.get("metrics", {})
            
            # Get metadata separately
            num_samples = results.get("num_samples", 0)
            
        except Exception as e:
            print(f"    ⚠️ Could not load results: {e}")
            task_metrics = {}
            num_samples = 0
        
        # -------- Add specificity if not already present --------
        if "specificity" not in task_metrics:
            tp = task_metrics.get("tp")
            fp = task_metrics.get("fp")
            tn = task_metrics.get("tn")
            fn = task_metrics.get("fn")
            
            if tn is not None and fp is not None and (tn + fp) > 0:
                task_metrics["specificity"] = tn / (tn + fp)
        
        # -------- Build row (WITHOUT predictions) --------
        row = {
            "model": model_name,
            **task_metrics,
            **collapse_metrics,
            "num_samples": num_samples
        }
        
        if dataset_name not in all_tables:
            all_tables[dataset_name] = []
        
        all_tables[dataset_name].append(row)

# ================================
# LOG ONE TABLE PER DATASET
# ================================

for dataset_name, rows in all_tables.items():
    print(f"\n📊 Logging table for dataset: {dataset_name}")
    
    # Check if ANY row has specificity
    has_specificity = any("specificity" in r for r in rows)
    
    # Format mean/std into "mean ± std"
    formatted_rows = []
    for r in rows:
        fr = {}
        for k, v in r.items():
            if k.endswith("_mean"):
                base = k.replace("_mean", "")
                std_key = base + "_std"
                if std_key in r:
                    fr[base] = fmt(r[k], r[std_key])
            elif k.endswith("_std"):
                continue
            else:
                fr[k] = v
        
        # Add specificity as None if this row doesn't have it but others do
        if has_specificity and "specificity" not in fr:
            fr["specificity"] = None
            
        formatted_rows.append(fr)
    
    columns = list(formatted_rows[0].keys())
    data = [[row.get(c, "nan") for c in columns] for row in formatted_rows]
    
    table = wandb.Table(columns=columns, data=data)
    wandb.log({f"table_{dataset_name}": table})

wandb.finish()

print("\n" + "="*80)
print("✅ META ANALYSIS COMPLETE!")
print("One table per dataset, rows = LoRA configs.")
print("="*80)