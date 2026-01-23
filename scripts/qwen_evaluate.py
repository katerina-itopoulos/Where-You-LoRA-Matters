# scripts/evaluate_and_extract_features.py
import os
os.environ['HF_HOME'] = '/home/ubuntu/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/home/ubuntu/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/home/ubuntu/.cache/huggingface/transformers'

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import gcsfs
from datasets import load_dataset, Dataset
from transformers import AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import json
import numpy as np
import gc
import shutil
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from src.experiments import run_pipeline_answers

# ================================
# CONFIG
# ================================

class EvalConfig:
    """Configuration for evaluation"""
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    GCS_BUCKET = "where_you_lora_matters_thesis"
    BATCH_SIZE = 1
    
    # Image resolution - same as training
    MIN_PIXELS = 196 * 32 * 32  # ~448x448 (same as training)
    MAX_PIXELS = 196 * 32 * 32
    
    # LoRA checkpoint (set to None for base model)
    LORA_CHECKPOINT = None
    IS_BASELINE = True 
    
    # Output directory
    EXPERIMENT_NAME = "baseline"  # Change for each experiment
    OUTPUT_DIR = f"experiments/{EXPERIMENT_NAME}"
    
    # VQAv2 split configuration (matching your training setup)
    VQAV2_TRAIN_SIZE = 20000
    VQAV2_VAL_SIZE = 2000
    # Test set = validation split, skipping train+val samples
    
    # Evaluation settings
    EVAL_VQAV2 = True
    EVAL_DASH_B = True
    EVAL_HALLUSIONBENCH = True
    EVAL_VQA_VS = True  # Set to True once VQA-VS is reprocessed
    
    # Sampling for quick tests (set to None for full eval)
    VQAV2_TEST_SAMPLE_SIZE = 100  # None for full test set (192,354 samples after skipping train+val)
    DASH_B_SAMPLE_SIZE = 100   # Already small dataset
    HALLUSION_SAMPLE_SIZE = 100  # Already small dataset
    VQA_VS_SAMPLE_SIZE = 100

config = EvalConfig()
fs = gcsfs.GCSFileSystem(token='google_default')

# ================================
# LOAD MODEL + LORA
# ================================

def load_model_and_processor(config, lora_checkpoint=None):
    """Load base model + LoRA adapter"""
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    print("\n[1/3] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
        min_pixels=config.MIN_PIXELS,
        max_pixels=config.MAX_PIXELS,
    )
    print("✓ Processor loaded")
    print(f"  Image resolution: ~448x448 (min/max pixels: {config.MIN_PIXELS:,})")
    
    print("\n[2/3] Loading base model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Disable gradient checkpointing for inference
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        print("✓ Gradient checkpointing disabled")
    
    print("✓ Base model loaded")
    
    # Load LoRA if checkpoint provided
    if lora_checkpoint:
        print(f"\n[3/3] Loading LoRA adapter from GCS...")
        local_adapter_path = "/tmp/lora_adapter"
        
        if os.path.exists(local_adapter_path):
            shutil.rmtree(local_adapter_path)
        os.makedirs(local_adapter_path)
        
        # Handle both directory and file paths
        if lora_checkpoint.endswith('.safetensors'):
            adapter_dir = lora_checkpoint.rsplit('/', 1)[0]
        else:
            adapter_dir = lora_checkpoint
        
        adapter_dir = adapter_dir if adapter_dir.startswith('gs://') else 'gs://' + adapter_dir
        adapter_dir_clean = adapter_dir.replace('gs://', '')
        
        files = fs.ls(adapter_dir_clean)
        adapter_files = [f for f in files if f.endswith(('.safetensors', '.json', '.bin'))]
        
        print(f"Downloading {len(adapter_files)} adapter files...")
        for file in adapter_files:
            filename = file.split('/')[-1]
            local_file = os.path.join(local_adapter_path, filename)
            fs.get(file, local_file)
        
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, local_adapter_path)
        
        # Merge LoRA weights into base model for faster inference
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        
        print("✓ LoRA adapter loaded and merged")
    else:
        print("\n[3/3] No LoRA checkpoint - using base model")
    
    model.eval()
    
    # Warmup to optimize CUDA kernels
    print("\nWarming up model...")
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    
    print(f"✓ Model ready on {device}")
    
    return model, processor

print("Loading model and processor...")
model, processor = load_model_and_processor(config, config.LORA_CHECKPOINT)

# --------------------------------
# Dataset-specific formatting
# --------------------------------

def format_dataset_for_pipeline(dataset, dataset_type):
    """
    Convert HF dataset to format expected by your pipeline
    Your run_batch_block_vectors expects rows with:
    - 'image': PIL Image
    - 'question': str
    - 'answers': list of answer dicts (VQAv2/VQA-VS format)
    - OR 'multiple_choice_answer': single answer string
    """
    formatted_rows = []
    
    for example in dataset:
        row = {
            'image': example['image'],
            'question': example['question'],
        }
        
        # Add answer based on dataset structure
        if dataset_type == "vqav2":
            # VQAv2 has 'answers' field with list of dicts
            row['answers'] = example.get('answers', [])
        
        elif dataset_type == "dash_b":
            # DASH-B has single 'answer' field (yes/no)
            row['multiple_choice_answer'] = example.get('answer', '')
        
        elif dataset_type == "hallusionbench":
            # HallusionBench has 'gt_answer' or 'answer' field
            answer = example.get('gt_answer', example.get('answer', ''))
            # Normalize to yes/no for consistency
            if str(answer) in ['1', 'true', 'True']:
                row['multiple_choice_answer'] = 'yes'
            elif str(answer) in ['0', 'false', 'False']:
                row['multiple_choice_answer'] = 'no'
            else:
                row['multiple_choice_answer'] = str(answer)
        
        elif dataset_type == "vqa_vs":
            # VQA-VS has same format as VQAv2 (now that we reprocessed it)
            row['answers'] = example.get('answers', [])
        
        # Keep metadata
        if 'question_id' in example:
            row['question_id'] = example['question_id']
        if 'image_id' in example:
            row['image_id'] = example['image_id']
        
        formatted_rows.append(row)
    
    return formatted_rows

def add_prompt_suffix(rows, dataset_type, is_baseline=False):
    """Add dataset-specific prompt suffixes"""
    if dataset_type == "dash_b":
        for row in rows:
            row['question'] = f"{row['question']} Answer with yes or no only."
    elif dataset_type == "hallusionbench":
        for row in rows:
            row['question'] = f"{row['question']} Answer with yes or no only."
    elif dataset_type in ["vqav2", "vqa_vs"] and is_baseline:  # Only for baseline
        for row in rows:
            row['question'] = f"{row['question']} Provide a short answer (one or a few words)."
    return rows

def parse_answer(answer, dataset_type):
    """Parse model output based on dataset type"""
    answer_lower = answer.lower().strip()
    
    if dataset_type in ["dash_b", "hallusionbench"]:
        # Both use yes/no format
        if "yes" in answer_lower or "1" in answer_lower or "true" in answer_lower:
            return "yes"
        elif "no" in answer_lower or "0" in answer_lower or "false" in answer_lower:
            return "no"
        return answer_lower
    
    else:  # vqav2, vqa-vs
        return answer.strip()

# --------------------------------
# Evaluation metrics
# --------------------------------

def compute_vqa_accuracy(predictions, references):
    """
    VQA-style accuracy (for VQAv2 and VQA-VS)
    Reference format: list of answer dicts [{"answer": "yes", ...}, ...]
    Formula: Acc(ans) = min(# humans that said ans / 3, 1)
    """
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        pred_clean = pred.lower().strip()
        
        # Count how many annotators gave each answer
        answer_counts = {}
        for ans_dict in ref:
            ans = ans_dict['answer'].lower().strip()
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
        
        # VQA accuracy: min(# humans that said answer / 3, 1)
        pred_count = answer_counts.get(pred_clean, 0)
        accuracy = min(pred_count / 3.0, 1.0)
        correct += accuracy
    
    return {"accuracy": correct / total if total > 0 else 0.0}

def compute_binary_classification_metrics(predictions, references, dataset_name=""):
    """
    Binary classification metrics for DASH-B and HallusionBench
    Computes: Accuracy, Precision, Recall, F1, Yes rate
    """
    tp = fp = tn = fn = 0
    
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower().strip()
        ref_lower = ref.lower().strip() if isinstance(ref, str) else str(ref).lower().strip()
        
        if pred_lower == "yes" and ref_lower == "yes":
            tp += 1
        elif pred_lower == "yes" and ref_lower == "no":
            fp += 1
        elif pred_lower == "no" and ref_lower == "no":
            tn += 1
        elif pred_lower == "no" and ref_lower == "yes":
            fn += 1
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    yes_rate = (tp + fp) / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'yes_rate': yes_rate,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }

# --------------------------------
# Unified evaluation + feature extraction
# --------------------------------

def evaluate_and_extract_features(dataset, dataset_name, dataset_type, metric_fn, output_dir, max_samples=None):
    """
    Unified pipeline: extract features AND generate answers in one pass
    Uses your run_pipeline_100 function
    """
    
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name}")
    print(f"{'='*70}")
    
    # Convert dataset to format expected by your pipeline
    print("Formatting dataset...")
    rows = format_dataset_for_pipeline(dataset, dataset_type)
    
    # Add dataset-specific prompts
    rows = add_prompt_suffix(rows, dataset_type, is_baseline=config.IS_BASELINE)
    
    # Limit samples if specified
    if max_samples:
        rows = rows[:max_samples]
        print(f"Sampling {max_samples} examples")
    
    # Convert to HF Dataset (required by your run_batch_block_vectors)
    ds = Dataset.from_list(rows)
    
    print(f"Running unified pipeline on {len(ds)} samples...")
    print("  - Extracting Vproj and T_fused vectors")
    print("  - Generating answers")
    
    # Use your run_pipeline_100 which does BOTH feature extraction and generation
    V0_all, Vproj_all, T_all, answers = run_pipeline_answers(
        model,
        processor,
        ds,
        batch_size=config.BATCH_SIZE,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        max_new_tokens=16,
    )
    
    print(f"✓ Extracted features: Vproj {Vproj_all.shape}, T_fused {T_all.shape}")
    print(f"✓ Generated {len(answers)} answers")
    
    # Parse answers based on dataset type
    predictions = [parse_answer(ans, dataset_type) for ans in answers]
    
    # Get references
    references = []
    for row in rows:
        if 'answers' in row:
            references.append(row['answers'])
        else:
            references.append(row['multiple_choice_answer'])
    
    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = metric_fn(predictions, references)
    
    print(f"\n📊 Results for {dataset_name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Save features to GCS
    features_path = f"{output_dir}/features/{dataset_name}_features.npz"
    gcs_features_path = f"gs://{config.GCS_BUCKET}/{features_path}"
    
    print(f"\nSaving features to {gcs_features_path}...")
    local_features_path = "/tmp/temp_features.npz"
    
    np.savez_compressed(
        local_features_path,
        Vproj=Vproj_all.float().cpu().numpy() if torch.is_tensor(Vproj_all) else Vproj_all,
        T_fused=T_all.float().cpu().numpy() if torch.is_tensor(T_all) else T_all,
        V0=V0_all.float().cpu().numpy() if V0_all is not None and torch.is_tensor(V0_all) else None,
    )
    
    with open(local_features_path, 'rb') as f:
        with fs.open(gcs_features_path, 'wb') as gcs_f:
            gcs_f.write(f.read())
    
    os.remove(local_features_path)
    print(f"✓ Features saved")
    
    # Save predictions and metrics
    results = {
        'dataset': dataset_name,
        'dataset_type': dataset_type,
        'metrics': metrics,
        'num_samples': len(predictions),
        'lora_checkpoint': config.LORA_CHECKPOINT,
        'experiment_name': config.EXPERIMENT_NAME,
        'predictions': [
            {
                'question_id': row.get('question_id'),
                'question': row['question'],
                'prediction': pred,
                'reference': ref,
            }
            for row, pred, ref in zip(rows, predictions, references)
        ]
    }
    
    results_path = f"{output_dir}/results/{dataset_name}_results.json"
    gcs_results_path = f"gs://{config.GCS_BUCKET}/{results_path}"
    
    print(f"Saving results to {gcs_results_path}...")
    local_results_path = "/tmp/temp_results.json"

    with open(local_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(local_results_path, 'rb') as f:
        with fs.open(gcs_results_path, 'wb') as gcs_f:
            gcs_f.write(f.read())
    
    os.remove(local_results_path)
    print(f"✓ Results saved")
    
    # Cleanup
    del ds, V0_all, Vproj_all, T_all, answers, predictions, references
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics

# --------------------------------
# Main evaluation
# --------------------------------

print("="*80)
print(f"EVALUATION: {config.EXPERIMENT_NAME}")
print("="*80)
print(f"LoRA checkpoint: {config.LORA_CHECKPOINT or 'None (base model)'}")
print(f"Output directory: gs://{config.GCS_BUCKET}/{config.OUTPUT_DIR}/")
print(f"\nDatasets to evaluate:")
print(f"  VQAv2: {config.EVAL_VQAV2}")
print(f"  DASH-B: {config.EVAL_DASH_B}")
print(f"  HallusionBench: {config.EVAL_HALLUSIONBENCH}")
print(f"  VQA-VS OOD: {config.EVAL_VQA_VS}")

all_results = {}

# ================================
# 1. VQAv2 Test (validation split, skipping train+val)
# ================================
if config.EVAL_VQAV2:
    print("\n" + "="*80)
    print("1️⃣ VQAv2 Test Set")
    print("="*80)
    print(f"Using validation split, skipping first {config.VQAV2_TRAIN_SIZE + config.VQAV2_VAL_SIZE} samples")
    
    vqav2_stream = load_dataset(
        "lmms-lab/VQAv2", 
        split="validation", 
        streaming=True, 
        cache_dir="/tmp/hf_datasets_cache"
    )
    
    # Skip train and val samples to get test set
    vqav2_test_stream = vqav2_stream.skip(config.VQAV2_TRAIN_SIZE + config.VQAV2_VAL_SIZE)
    
    # Convert streaming to list (with optional sampling)
    if config.VQAV2_TEST_SAMPLE_SIZE:
        print(f"Sampling {config.VQAV2_TEST_SAMPLE_SIZE} test samples...")
        vqav2_list = list(vqav2_test_stream.take(config.VQAV2_TEST_SAMPLE_SIZE))
    else:
        print("Loading full VQAv2 test set (192,354 samples - this may take a while)...")
        vqav2_list = list(vqav2_test_stream)
    
    print(f"Loaded {len(vqav2_list)} test samples")
    vqav2_ds = Dataset.from_list(vqav2_list)
    
    all_results['vqav2_test'] = evaluate_and_extract_features(
        vqav2_ds, 
        "vqav2_test", 
        "vqav2", 
        compute_vqa_accuracy,
        config.OUTPUT_DIR
    )
    
    del vqav2_stream, vqav2_test_stream, vqav2_list, vqav2_ds
    gc.collect()
    torch.cuda.empty_cache()

# ================================
# 2. DASH-B
# ================================
if config.EVAL_DASH_B:
    print("\n" + "="*80)
    print("DASH-B Dataset")
    print("="*80)
    
    try:
        dashb_data = load_dataset(
            "YanNeu/DASH-B",  # Adjust if different
            split="test", 
            cache_dir="/tmp/hf_datasets_cache"
        )
        
        all_results['dash_b'] = evaluate_and_extract_features(
            dashb_data,
            "dash_b",
            "dash_b",
            lambda p, r: compute_binary_classification_metrics(p, r, "DASH-B"),
            config.OUTPUT_DIR,
            max_samples=config.DASH_B_SAMPLE_SIZE
        )
        
        del dashb_data
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f" Error loading DASH-B: {e}")
        print("Please verify the correct HuggingFace dataset path for DASH-B")
        print("You may need to download it manually or check the dataset name")

# ================================
# 3. HallusionBench
# ================================
if config.EVAL_HALLUSIONBENCH:
    print("\n" + "="*80)
    print("3️⃣ HallusionBench")
    print("="*80)
    
    # Process both splits
    for split_name in ["image", "non_image"]:
        print(f"\n  Processing HallusionBench {split_name} split...")
        
        try:
            hallusion_data = load_dataset(
                "lmms-lab/HallusionBench", 
                split=split_name,  # Use 'image' or 'non_image'
                cache_dir=None
            )
            
            all_results[f'hallusionbench_{split_name}'] = evaluate_and_extract_features(
                hallusion_data,
                f"hallusionbench_{split_name}",
                "hallusionbench",
                lambda p, r: compute_binary_classification_metrics(p, r, f"HallusionBench-{split_name}"),
                config.OUTPUT_DIR,
                max_samples=config.HALLUSION_SAMPLE_SIZE
            )
            
            del hallusion_data
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"⚠️ Error loading HallusionBench {split_name}: {e}")
            import traceback
            traceback.print_exc()

# ================================
# 4. VQA-VS OOD sets
# ================================
# In the VQA-VS evaluation section (around line 490), update:

if config.EVAL_VQA_VS:
    print("\n" + "="*80)
    print("4️⃣ VQA-VS OOD Sets")
    print("="*80)
    
    OOD_CATEGORIES = ['ko', 'kop', 'kw_ko', 'kw', 'kwp', 'qt_ko', 'qt_kw_ko', 'qt_kw', 'qt']
    
    for category in OOD_CATEGORIES:
        print(f"\n  Processing OOD {category}...")
        
        ood_path = f"gs://{config.GCS_BUCKET}/datasets/vqa-vs-preprocessed/ood_{category}/*.parquet"
        
        try:
            ood_data = load_dataset("parquet", data_files=ood_path, split="train")
            
            # Sample if specified
            if hasattr(config, 'VQA_VS_SAMPLE_SIZE') and config.VQA_VS_SAMPLE_SIZE:
                print(f"  Sampling {config.VQA_VS_SAMPLE_SIZE} from {len(ood_data)} samples")
                import random
                indices = random.sample(range(len(ood_data)), min(config.VQA_VS_SAMPLE_SIZE, len(ood_data)))
                ood_data = ood_data.select(indices)
            
            all_results[f'vqa_vs_ood_{category}'] = evaluate_and_extract_features(
                ood_data,
                f"vqa_vs_ood_{category}",
                "vqa_vs",
                compute_vqa_accuracy,
                config.OUTPUT_DIR
            )
            
            del ood_data
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"⚠️ Error loading {category}: {e}")
            import traceback
            traceback.print_exc()

# ================================
# Save summary
# ================================
print("\n" + "="*80)
print("ALL PROCESSING COMPLETE!")
print("="*80)

summary = {
    'experiment_name': config.EXPERIMENT_NAME,
    'lora_checkpoint': config.LORA_CHECKPOINT,
    'model_name': config.MODEL_NAME,
    'batch_size': config.BATCH_SIZE,
    'vqav2_split_info': {
        'train_size': config.VQAV2_TRAIN_SIZE,
        'val_size': config.VQAV2_VAL_SIZE,
        'test_size': len(all_results.get('vqav2_test', {}).get('predictions', [])) if 'vqav2_test' in all_results else 0
    },
    'results': all_results
}

summary_path = f"{config.OUTPUT_DIR}/evaluation_summary.json"
gcs_summary_path = f"gs://{config.GCS_BUCKET}/{summary_path}"

with open("/tmp/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

with open("/tmp/summary.json", 'rb') as f:
    with fs.open(gcs_summary_path, 'wb') as gcs_f:
        gcs_f.write(f.read())

os.remove("/tmp/summary.json")
print(f"\nSummary saved to {gcs_summary_path}")

print("\nSummary of all results:")
for dataset, metrics in all_results.items():
    print(f"\n{dataset}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

print("\n All outputs saved to:")
print(f"  Features: gs://{config.GCS_BUCKET}/{config.OUTPUT_DIR}/features/")
print(f"  Results:  gs://{config.GCS_BUCKET}/{config.OUTPUT_DIR}/results/")
print(f"  Summary:  {gcs_summary_path}")


