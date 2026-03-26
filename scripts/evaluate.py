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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import json
import numpy as np
import gc
import shutil
import pandas as pd
from src.experiments import run_pipeline_answers


class EvalConfig:
    """Evaluation configuration for all benchmarks."""
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    GCS_BUCKET = "where_you_lora_matters_thesis"
    BATCH_SIZE = 1
    
    MIN_PIXELS = 196 * 32 * 32
    MAX_PIXELS = 196 * 32 * 32
    
    LORA_CHECKPOINT = None
    IS_BASELINE = True
    
    EXPERIMENT_NAME = "baseline"
    OUTPUT_DIR = f"experiments/{EXPERIMENT_NAME}"
    
    VQAV2_TRAIN_SIZE = 20000
    VQAV2_VAL_SIZE = 2000
    
    EVAL_VQAV2 = True
    EVAL_DASH_B = True
    EVAL_HALLUSIONBENCH = True
    EVAL_MMVP = True
    EVAL_VQA_VS = True
    
    VQAV2_TEST_SAMPLE_SIZE = 2000
    DASH_B_SAMPLE_SIZE = None
    HALLUSION_SAMPLE_SIZE = None
    MMVP_SAMPLE_SIZE = None
    VQA_VS_SAMPLE_SIZE = 2000


config = EvalConfig()
fs = gcsfs.GCSFileSystem(token='google_default')


def load_model_and_processor(config, lora_checkpoint=None):
    """Load base model and optional LoRA adapter."""
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
    print(f"Processor loaded (resolution: ~448x448, {config.MIN_PIXELS:,} pixels)")
    
    print("\n[2/3] Loading base model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    if lora_checkpoint:
        print(f"\n[3/3] Loading LoRA adapter from GCS...")
        local_adapter_path = "/tmp/lora_adapter"
        
        if os.path.exists(local_adapter_path):
            shutil.rmtree(local_adapter_path)
        os.makedirs(local_adapter_path)
        
        adapter_dir = lora_checkpoint.rsplit('/', 1)[0] if lora_checkpoint.endswith('.safetensors') else lora_checkpoint
        adapter_dir = adapter_dir if adapter_dir.startswith('gs://') else 'gs://' + adapter_dir
        adapter_dir_clean = adapter_dir.replace('gs://', '')
        
        files = fs.ls(adapter_dir_clean)
        adapter_files = [f for f in files if f.endswith(('.safetensors', '.json', '.bin'))]
        
        print(f"Downloading {len(adapter_files)} adapter files...")
        for file in adapter_files:
            filename = file.split('/')[-1]
            local_file = os.path.join(local_adapter_path, filename)
            fs.get(file, local_file)
        
        model = PeftModel.from_pretrained(model, local_adapter_path)
        model = model.merge_and_unload()
        print("LoRA adapter loaded and merged")
    else:
        print("\n[3/3] Using base model (no LoRA)")
    
    model.eval()
    torch.cuda.empty_cache()
    
    device = next(model.parameters()).device
    print(f"Model ready on {device}")
    
    return model, processor


print("Loading model and processor...")
model, processor = load_model_and_processor(config, config.LORA_CHECKPOINT)


def format_dataset_for_pipeline(dataset, dataset_type):
    """Convert HF dataset to pipeline format."""
    formatted_rows = []
    
    for example in dataset:
        row = {
            'image': example['image'],
            'question': example['question'],
        }
        
        if dataset_type == "vqav2":
            row['answers'] = example.get('answers', [])
        elif dataset_type == "dash_b":
            row['multiple_choice_answer'] = example.get('answer', '')
        elif dataset_type == "hallusionbench":
            answer = example.get('gt_answer', example.get('answer', ''))
            if str(answer) in ['1', 'true', 'True']:
                row['multiple_choice_answer'] = 'yes'
            elif str(answer) in ['0', 'false', 'False']:
                row['multiple_choice_answer'] = 'no'
            else:
                row['multiple_choice_answer'] = str(answer)
        elif dataset_type == "vqa_vs":
            row['answers'] = example.get('answers', [])
        
        if 'question_id' in example:
            row['question_id'] = example['question_id']
        if 'image_id' in example:
            row['image_id'] = example['image_id']
        
        formatted_rows.append(row)
    
    return formatted_rows


def add_prompt_suffix(rows, dataset_type, is_baseline=False):
    """Add dataset-specific prompt suffixes."""
    if dataset_type == "dash_b":
        for row in rows:
            row['question'] = f"{row['question']} Answer with yes or no only."
    elif dataset_type == "hallusionbench":
        for row in rows:
            row['question'] = f"{row['question']} Answer with yes or no only."
    elif dataset_type in ["vqav2", "vqa_vs"] and is_baseline:
        for row in rows:
            row['question'] = f"{row['question']} Provide a short answer (one or a few words)."
    return rows


def parse_answer(answer, dataset_type):
    """Parse model output based on dataset type."""
    answer_lower = answer.lower().strip()
    
    if dataset_type in ["dash_b", "hallusionbench"]:
        if "yes" in answer_lower or "1" in answer_lower or "true" in answer_lower:
            return "yes"
        elif "no" in answer_lower or "0" in answer_lower or "false" in answer_lower:
            return "no"
        return answer_lower
    else:
        return answer.strip()


def compute_vqa_accuracy(predictions, references):
    """VQA-style soft accuracy: Acc(ans) = min(#humans/3, 1)."""
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        pred_clean = pred.lower().strip()
        
        answer_counts = {}
        for ans_dict in ref:
            ans = ans_dict['answer'].lower().strip()
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
        
        pred_count = answer_counts.get(pred_clean, 0)
        accuracy = min(pred_count / 3.0, 1.0)
        correct += accuracy
    
    return {"accuracy": correct / total if total > 0 else 0.0}


def compute_binary_classification_metrics(predictions, references, dataset_name=""):
    """Binary classification metrics: accuracy, precision, recall, F1."""
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


def evaluate_and_extract_features(dataset, dataset_name, dataset_type, metric_fn, output_dir, max_samples=None):
    """Run inference, extract features, and compute metrics."""
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name}")
    print(f"{'='*70}")
    
    print("Formatting dataset...")
    rows = format_dataset_for_pipeline(dataset, dataset_type)
    rows = add_prompt_suffix(rows, dataset_type, is_baseline=config.IS_BASELINE)
    
    if max_samples:
        rows = rows[:max_samples]
        print(f"Sampling {max_samples} examples")
    
    ds = Dataset.from_list(rows)
    
    print(f"Running pipeline on {len(ds)} samples...")
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
    
    print(f"Extracted features: Vproj {Vproj_all.shape}, T_fused {T_all.shape}")
    print(f"Generated {len(answers)} answers")
    
    predictions = [parse_answer(ans, dataset_type) for ans in answers]
    
    references = []
    for row in rows:
        if 'answers' in row:
            references.append(row['answers'])
        else:
            references.append(row['multiple_choice_answer'])
    
    metrics = metric_fn(predictions, references)
    
    print(f"\nResults for {dataset_name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    features_path = f"{output_dir}/features/{dataset_name}_features.npz"
    gcs_features_path = f"gs://{config.GCS_BUCKET}/{features_path}"
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
    local_results_path = "/tmp/temp_results.json"

    with open(local_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(local_results_path, 'rb') as f:
        with fs.open(gcs_results_path, 'wb') as gcs_f:
            gcs_f.write(f.read())
    os.remove(local_results_path)
    
    del ds, V0_all, Vproj_all, T_all, answers, predictions, references
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics


print("="*80)
print(f"EVALUATION: {config.EXPERIMENT_NAME}")
print("="*80)
print(f"LoRA checkpoint: {config.LORA_CHECKPOINT or 'None (base model)'}")
print(f"Output directory: gs://{config.GCS_BUCKET}/{config.OUTPUT_DIR}/")

all_results = {}


if config.EVAL_VQAV2:
    print("\n" + "="*80)
    print("VQAv2 Test Set")
    print("="*80)
    
    vqav2_stream = load_dataset(
        "lmms-lab/VQAv2", 
        split="validation", 
        streaming=True, 
        cache_dir="/tmp/hf_datasets_cache"
    )
    
    vqav2_test_stream = vqav2_stream.skip(config.VQAV2_TRAIN_SIZE + config.VQAV2_VAL_SIZE)
    
    if config.VQAV2_TEST_SAMPLE_SIZE:
        vqav2_list = list(vqav2_test_stream.take(config.VQAV2_TEST_SAMPLE_SIZE))
    else:
        vqav2_list = list(vqav2_test_stream)
    
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


if config.EVAL_DASH_B:
    print("\n" + "="*80)
    print("DASH-B Dataset")
    print("="*80)
    
    try:
        dashb_data = load_dataset("YanNeu/DASH-B", split="test", cache_dir="/tmp/hf_datasets_cache")
        
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
        print(f"Error loading DASH-B: {e}")


if config.EVAL_HALLUSIONBENCH:
    print("\n" + "="*80)
    print("HallusionBench")
    print("="*80)
    
    for split_name in ["image"]:
        try:
            hallusion_data = load_dataset("lmms-lab/HallusionBench", split=split_name, cache_dir=None)
            
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
            print(f"Error loading HallusionBench {split_name}: {e}")


if config.EVAL_MMVP:
    print("\n" + "="*80)
    print("MMVP (Multi-Modal Visual Patterns)")
    print("="*80)
    
    try:
        img_ds = load_dataset("MMVP/MMVP", split="train", cache_dir="/tmp/hf_datasets_cache")
        
        photo_id_to_img = {}
        for i in range(len(img_ds)):
            img = img_ds[i]["image"]
            fname = os.path.basename(img.filename)
            photo_id = int(fname.replace(".jpg", ""))
            photo_id_to_img[photo_id] = img
        
        csv_path = f"gs://{config.GCS_BUCKET}/datasets/mmvp/Questions.csv"
        df = pd.read_csv(csv_path)
        
        rows = []
        for i in range(len(df)):
            photo_id = i + 1
            rows.append({
                "image": photo_id_to_img[photo_id],
                "question": df.iloc[i]["Question"],
                "options": df.iloc[i]["Options"],
                "correct_answer": str(df.iloc[i]["Correct Answer"]).strip(),
            })
        
        mmvp_ds = Dataset.from_list(rows)
        
        def parse_answer_mmvp(answer):
            a = (answer or "").lower().strip()
            if "(a)" in a or a == "a":
                return "(a)"
            if "(b)" in a or a == "b":
                return "(b)"
            return a
        
        def compute_mmvp_accuracy(predictions, references):
            correct = sum(int(p == r) for p, r in zip(predictions, references))
            n = len(predictions)
            return {"accuracy": (correct / n) if n > 0 else 0.0, "num_samples": n}
        
        formatted_rows = []
        for example in mmvp_ds:
            row = {
                'image': example['image'],
                'question': f"{example['question']}\nOptions:\n{example['options']}\nYour answer must be exactly one of: (a) or (b).",
                'multiple_choice_answer': example['correct_answer']
            }
            formatted_rows.append(row)
        
        if config.MMVP_SAMPLE_SIZE:
            formatted_rows = formatted_rows[:config.MMVP_SAMPLE_SIZE]
        
        mmvp_ds_formatted = Dataset.from_list(formatted_rows)
        
        V0_all, Vproj_all, T_all, answers = run_pipeline_answers(
            model, processor, mmvp_ds_formatted,
            batch_size=config.BATCH_SIZE,
            do_sample=False, temperature=0.0, top_p=1.0, top_k=1,
            repetition_penalty=1.0, max_new_tokens=16,
        )
        
        predictions = [parse_answer_mmvp(ans) for ans in answers]
        references = [row['multiple_choice_answer'] for row in formatted_rows]
        metrics = compute_mmvp_accuracy(predictions, references)
        
        print(f"\nMMVP Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        features_path = f"{config.OUTPUT_DIR}/features/mmvp_features.npz"
        gcs_features_path = f"gs://{config.GCS_BUCKET}/{features_path}"
        local_features_path = "/tmp/temp_features.npz"
        
        np.savez_compressed(
            local_features_path,
            Vproj=Vproj_all.float().cpu().numpy(),
            T_fused=T_all.float().cpu().numpy(),
            V0=V0_all.float().cpu().numpy() if V0_all is not None else None,
        )
        
        with open(local_features_path, 'rb') as f:
            with fs.open(gcs_features_path, 'wb') as gcs_f:
                gcs_f.write(f.read())
        os.remove(local_features_path)
        
        results = {
            'dataset': 'mmvp',
            'dataset_type': 'mmvp',
            'metrics': metrics,
            'num_samples': len(predictions),
            'lora_checkpoint': config.LORA_CHECKPOINT,
            'experiment_name': config.EXPERIMENT_NAME,
            'predictions': [
                {'question': row['question'], 'prediction': pred, 'reference': ref}
                for row, pred, ref in zip(formatted_rows, predictions, references)
            ]
        }
        
        results_path = f"{config.OUTPUT_DIR}/results/mmvp_results.json"
        gcs_results_path = f"gs://{config.GCS_BUCKET}/{results_path}"
        local_results_path = "/tmp/temp_results.json"
        
        with open(local_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        with open(local_results_path, 'rb') as f:
            with fs.open(gcs_results_path, 'wb') as gcs_f:
                gcs_f.write(f.read())
        os.remove(local_results_path)
        
        all_results['mmvp'] = metrics
        
        del img_ds, mmvp_ds, mmvp_ds_formatted, V0_all, Vproj_all, T_all, answers
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error loading MMVP: {e}")


if config.EVAL_VQA_VS:
    print("\n" + "="*80)
    print("VQA-VS OOD Sets")
    print("="*80)
    
    OOD_CATEGORIES = ['ko', 'kop', 'kw_ko', 'kw', 'kwp', 'qt_ko', 'qt_kw_ko', 'qt_kw', 'qt']
    
    for category in OOD_CATEGORIES:
        ood_path = f"gs://{config.GCS_BUCKET}/datasets/vqa-vs-preprocessed/ood_{category}/*.parquet"
        
        try:
            ood_data = load_dataset("parquet", data_files=ood_path, split="train")
            
            if config.VQA_VS_SAMPLE_SIZE:
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
            print(f"Error loading {category}: {e}")


print("\n" + "="*80)
print("EVALUATION COMPLETE")
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
print("\nResults summary:")
for dataset, metrics in all_results.items():
    print(f"\n{dataset}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

print(f"\nOutputs saved to:")
print(f"  Features: gs://{config.GCS_BUCKET}/{config.OUTPUT_DIR}/features/")
print(f"  Results:  gs://{config.GCS_BUCKET}/{config.OUTPUT_DIR}/results/")
print(f"  Summary:  {gcs_summary_path}")