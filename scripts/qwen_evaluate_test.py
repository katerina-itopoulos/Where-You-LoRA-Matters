# scripts/qwen_evaluate_test.py

"""
Test Evaluation Run - LLM Only LoRA
====================================
Evaluation using preprocessed VQAv2 data
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import gcsfs
from datasets import Dataset, load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as fs
import json
from datetime import datetime
import gc

from src.collapse_metrics import summarize_vectors
from src.inference_utils import _coerce_visual_tokens

# ================================
# DATA LOADING & COLLATION
# ================================

def load_parquet_shards_from_gcs(bucket, prefix, split, num_samples=None):
    """
    Load parquet shards from GCS into a HuggingFace Dataset.
    
    Args:
        bucket: GCS bucket name
        prefix: Path prefix (e.g., "datasets/preprocessed_vqa_lowres")
        split: "train", "val", or "test"
        num_samples: Optional - take only first N samples
    """
    gcs = fs.GcsFileSystem()
    shard_path = f"{bucket}/{prefix}/{split}/"
    
    # List all parquet files
    file_info = gcs.get_file_info(fs.FileSelector(shard_path, recursive=False))
    parquet_files = [f.path for f in file_info if f.path.endswith('.parquet')]
    parquet_files.sort()  # Ensure consistent ordering
    
    print(f"Found {len(parquet_files)} shards for {split}")
    
    # Read all shards into one table
    tables = []
    total_rows = 0
    
    for filepath in parquet_files:
        table = pq.read_table(f"gs://{filepath}")
        
        if num_samples is not None:
            remaining = num_samples - total_rows
            if remaining <= 0:
                break
            if len(table) > remaining:
                table = table.slice(0, remaining)
        
        tables.append(table)
        total_rows += len(table)
        
        if num_samples is not None and total_rows >= num_samples:
            break
    
    # Concatenate all tables
    full_table = pa.concat_tables(tables)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset(full_table)
    
    print(f"✓ Loaded {len(dataset)} samples for {split}")
    return dataset

class VLDataCollatorPadTorch:
    def __call__(self, features):
        input_ids = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.as_tensor(f["attention_mask"], dtype=torch.long) for f in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # labels: only add if they're token IDs (lists/tensors), not strings
        if "labels" in features[0]:
            first_label = features[0]["labels"]
            # Check if labels are token IDs (list/array) or strings (answers)
            if isinstance(first_label, (list, tuple)) or hasattr(first_label, '__iter__') and not isinstance(first_label, str):
                labels = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]
                labels = pad_sequence(labels, batch_first=True, padding_value=-100)
                labels = labels[:, : input_ids.shape[1]]
                batch["labels"] = labels
            # else: labels are strings (answers), skip them for inference

        # vision
        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            pv_list = [torch.as_tensor(f["pixel_values"]) for f in features]
            batch["pixel_values"] = torch.cat(pv_list, dim=0)
            batch["image_grid_thw"] = torch.stack(
                [torch.as_tensor(f["image_grid_thw"], dtype=torch.long) for f in features], dim=0
            )

        return batch

# ================================
# CONFIG
# ================================

class TestConfig:
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    
    GCS_BUCKET = "where_you_lora_matters_thesis"
    DATA_PREFIX = "datasets/preprocessed_vqa_lowres"
    OUTPUT_DIR = "evaluation_results"
    
    LORA_CHECKPOINT = "gs://where_you_lora_matters_thesis/artifacts/Qwen3-VL/projector_only/checkpoint-6000"
    
    BATCH_SIZE = 4
    MAX_SAMPLES = 20
    
    MAX_NEW_TOKENS = 50

print("Configuration loaded ✓")

# ================================
# LOAD MODEL
# ================================

def load_model_and_processor(config, lora_checkpoint=None):
    """Load base model + LoRA adapter"""
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    print("\n[1/3] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True
    )
    print("✓ Processor loaded")
    
    print("\n[2/3] Loading base model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ Base model loaded")
    
    # Load LoRA if checkpoint provided
    if lora_checkpoint:
        print(f"\n[3/3] Loading LoRA adapter from GCS...")
        fs_gcs = gcsfs.GCSFileSystem(token='google_default')
        local_adapter_path = "/tmp/lora_adapter"
        
        import os
        import shutil
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
        
        files = fs_gcs.ls(adapter_dir_clean)
        adapter_files = [f for f in files if f.endswith(('.safetensors', '.json', '.bin'))]
        
        print(f"Downloading {len(adapter_files)} adapter files...")
        for file in adapter_files:
            filename = file.split('/')[-1]
            local_file = os.path.join(local_adapter_path, filename)
            fs_gcs.get(file, local_file)
        
        model = PeftModel.from_pretrained(model, local_adapter_path)
        model.eval()
        print("✓ LoRA adapter loaded")
    else:
        print("\n[3/3] No LoRA checkpoint - using base model")
        model.eval()
    
    device = next(model.parameters()).device
    print(f"✓ Model ready on {device}")
    
    return model, processor

# ================================
# LOAD DATASET
# ================================

def load_test_dataset(config):
    """Load preprocessed VQAv2 test data using your function"""
    print("\n" + "="*80)
    print("LOADING TEST DATASET")
    print("="*80)
    
    dataset = load_parquet_shards_from_gcs(
        bucket=config.GCS_BUCKET,
        prefix=config.DATA_PREFIX,
        split="test",
        num_samples=config.MAX_SAMPLES
    )
    
    return dataset

# ================================
# EVALUATION
# ================================
def extract_features_from_batch(model, batch):
    """
    Extract features by calling model.visual() directly on preprocessed pixel_values
    """
    try:
        with torch.no_grad():
            # Get LLM hidden states
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            
            H = outputs.hidden_states[-1]  # [B, seq_len, D_llm]
            B, seq_len, D_llm = H.shape
            
            # Call model.visual()
            print(f"    👁️ Calling model.visual()...")
            Vraw = model.visual(
                batch['pixel_values'],
                grid_thw=batch['image_grid_thw']
            )
            
            # Extract vision features from tuple
            if isinstance(Vraw, tuple):
                vision_flat = Vraw[0]  # [total_tokens, D]
                print(f"    📦 Vision features (flat): {vision_flat.shape}")
                
                # Calculate tokens per image from grid
                grid = batch['image_grid_thw']  # [B, 3] where each is [T, H, W]
                # For images: T=1, tokens_per_image = H * W
                T, H_grid, W_grid = grid[0]  # First sample's grid
                Ti = int(H_grid * W_grid)
                print(f"    📊 Grid: T={T}, H={H_grid}, W={W_grid} → Ti={Ti} tokens/image")
                
                # Reshape: [total_tokens, D] → [B, Ti, D]
                total_tokens = vision_flat.shape[0]
                expected_total = B * Ti
                
                if total_tokens != expected_total:
                    print(f"    ⚠️ Token mismatch: got {total_tokens}, expected {expected_total}")
                    # Try to infer Ti
                    Ti = total_tokens // B
                    print(f"    🔧 Adjusting Ti to {Ti}")
                
                Vproj = vision_flat.view(B, Ti, D_llm)
                print(f"    ✓ Reshaped Vproj: {Vproj.shape}")
                
            elif torch.is_tensor(Vraw):
                Vproj = Vraw
                print(f"    📦 Direct tensor: {Vproj.shape}")
            else:
                print(f"    ❌ Unexpected type: {type(Vraw)}")
                return None, None
            
            # Pool vision tokens
            Vproj_vec = Vproj.mean(dim=1).cpu()
            
            # Get text tokens (skip first Ti image tokens)
            Ti = int(Vproj.shape[1])
            print(f"    📊 Ti={Ti}, seq_len={seq_len}")
            T_span = H[:, Ti:, :] if Ti < seq_len else H
            T_fused_vec = T_span.mean(dim=1).cpu()
            
            print(f"    ✓ Vproj_vec: {Vproj_vec.shape}, T_fused_vec: {T_fused_vec.shape}")
            return Vproj_vec, T_fused_vec
            
    except Exception as e:
        print(f"    ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_evaluation(model, processor, dataset, config):
    """Evaluate on preprocessed data with proper feature extraction"""
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    
    def collate_batch(samples, device):
        """Your working collator from VQAAccuracyCallback"""
        input_ids = [torch.as_tensor(s['input_ids']) for s in samples]
        attention_mask = [torch.as_tensor(s['attention_mask']) for s in samples]

        pad_token_id = processor.tokenizer.pad_token_id or 0
        max_len = max(x.size(0) for x in input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - ids.size(0)
            padded_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_token_id), ids])
            )
            padded_attention_masks.append(
                torch.cat([torch.zeros(pad_len), mask])
            )

        input_ids_padded = torch.stack(padded_input_ids).to(device)
        attention_mask_padded = torch.stack(padded_attention_masks).to(device)

        pixel_values = torch.cat(
            [torch.as_tensor(s['pixel_values']) for s in samples], dim=0
        ).to(device)

        image_grid_thw = torch.stack(
            [torch.as_tensor(s['image_grid_thw']) for s in samples], dim=0
        ).to(device)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
    
    results = []
    all_features = {
        'Vproj_vecs': [],
        'T_fused_vecs': [],
        'question_ids': []
    }
    
    num_samples = len(dataset)
    num_batches = (num_samples + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    print(f"\nProcessing {num_samples} samples in {num_batches} batches\n")
    
    # Set left padding
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.BATCH_SIZE
            end_idx = min(start_idx + config.BATCH_SIZE, num_samples)
            
            print(f"Batch {batch_idx+1}/{num_batches} [{start_idx}:{end_idx}]")
            
            batch_samples = [dataset[i] for i in range(start_idx, end_idx)]
            answers = [s['answer'] for s in batch_samples]
            
            try:
                # Collate batch
                batch = collate_batch(batch_samples, model.device)
                input_len = batch['input_ids'].shape[1]
                
                # Extract features (using your proven method!)
                print(f"  🔍 Extracting features...")
                Vproj_vec, T_fused_vec = extract_features_from_batch(model, batch)
                
                # DEBUG: Check what we got
                print(f"  📊 Feature extraction result:")
                print(f"     - Vproj_vec: {Vproj_vec is not None}")
                if Vproj_vec is not None:
                    print(f"       Shape: {Vproj_vec.shape}")
                print(f"     - T_fused_vec: {T_fused_vec is not None}")
                if T_fused_vec is not None:
                    print(f"       Shape: {T_fused_vec.shape}")
                
                # Generate predictions
                generated_ids = model.generate(
                    **batch,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                
                # Decode
                for i in range(len(batch_samples)):
                    prediction = processor.tokenizer.decode(
                        generated_ids[i][input_len:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    qid = f"vqav2_test_{start_idx + i}"
                    
                    result = {
                        'question_id': qid,
                        'prediction': prediction,
                        'ground_truth': answers[i],
                    }
                    results.append(result)
                    
                    match = prediction.lower() == answers[i].lower()
                    print(f"  [{i+1}] {'✓' if match else '✗'} Pred: '{prediction}' | GT: '{answers[i]}'")
                
                # Store features (if extraction succeeded)
                if Vproj_vec is not None and T_fused_vec is not None:
                    print(f"  ✅ Storing features for batch")
                    for i in range(len(batch_samples)):
                        all_features['Vproj_vecs'].append(Vproj_vec[i:i+1])
                        all_features['T_fused_vecs'].append(T_fused_vec[i:i+1])
                        all_features['question_ids'].append(f"vqav2_test_{start_idx + i}")
                else:
                    print(f"  ⚠️ Skipping feature storage (extraction failed)")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            gc.collect()
            torch.cuda.empty_cache()
    
    # Restore padding side
    processor.tokenizer.padding_side = original_padding_side
    
    # DEBUG: Feature collection summary
    print(f"\n{'='*80}")
    print("FEATURE COLLECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Vproj_vecs collected: {len(all_features['Vproj_vecs'])}")
    print(f"T_fused_vecs collected: {len(all_features['T_fused_vecs'])}")
    print(f"Question IDs collected: {len(all_features['question_ids'])}")
    
    # Stack features
    if all_features['Vproj_vecs']:
        all_features['Vproj_vecs'] = torch.cat(all_features['Vproj_vecs'], dim=0)
        all_features['T_fused_vecs'] = torch.cat(all_features['T_fused_vecs'], dim=0)
        print(f"\n✓ Features stacked:")
        print(f"  - Vproj final shape: {all_features['Vproj_vecs'].shape}")
        print(f"  - T_fused final shape: {all_features['T_fused_vecs'].shape}")
    else:
        print(f"\n⚠️ No features collected! Setting features to None")
        all_features = None
    print(f"{'='*80}\n")
    
    return results, all_features

# ================================
# METRICS
# ================================

def compute_metrics(results, features):
    """Compute all metrics including collapse metrics"""
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)
    
    # 1. Task accuracy
    correct = sum(
        1 for r in results 
        if r['prediction'].lower().strip() == r['ground_truth'].lower().strip()
    )
    accuracy = correct / len(results) if results else 0.0
    
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(results)})")
    
    # 2. Collapse metrics
    collapse_metrics = {}
    if features and features.get('Vproj_vecs') is not None:
        print("\nComputing collapse metrics...")
        
        collapse_metrics = summarize_vectors(
            V0_all=None,  # We don't have raw vision features
            Vproj_all=features['Vproj_vecs'],
            T_all=features['T_fused_vecs']
        )
        
        print("\n✓ Collapse Metrics:")
        for k, v in collapse_metrics.items():
            if isinstance(v, float) and v == v:  # Not NaN
                print(f"  {k}: {v:.4f}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'num_samples': len(results),
        **collapse_metrics
    }

# ================================
# SAVE
# ================================
def save_results(results, features, metrics, config, experiment_name, dataset_name):
    """Save results, features, and metrics to GCS"""
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Debug: Check what we received
    print(f"📊 Saving data:")
    print(f"  - Results: {len(results)} predictions")
    print(f"  - Features: {features is not None}")
    if features:
        print(f"    - Vproj: {features.get('Vproj_vecs') is not None}")
        print(f"    - T_fused: {features.get('T_fused_vecs') is not None}")
        print(f"    - Question IDs: {len(features.get('question_ids', []))}")
    print(f"  - Metrics: {len(metrics)} keys")
    
    fs = gcsfs.GCSFileSystem(token='google_default')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{config.GCS_BUCKET}/{config.OUTPUT_DIR}/{experiment_name}/{dataset_name}_{timestamp}"
    
    print(f"\n📁 Output directory: gs://{output_dir}")
    
    try:
        # 1. Save predictions
        predictions_path = f"gs://{output_dir}/predictions.jsonl"
        print(f"\n💾 Saving predictions to {predictions_path}")
        with fs.open(predictions_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"✓ Predictions saved ({len(results)} lines)")
        
        # 2. Save features
        if features is not None and features.get('Vproj_vecs') is not None:
            features_dict = {
                'Vproj_vecs': features['Vproj_vecs'],
                'T_fused_vecs': features['T_fused_vecs'],
                'question_ids': features['question_ids'],
                'metadata': {
                    'experiment_name': experiment_name,
                    'dataset_name': dataset_name,
                    'timestamp': timestamp,
                    'num_samples': len(features['question_ids']),
                    'feature_dim': features['Vproj_vecs'].shape[-1],
                    'shape_Vproj': list(features['Vproj_vecs'].shape),
                    'shape_T_fused': list(features['T_fused_vecs'].shape),
                }
            }
            
            features_path = f"gs://{output_dir}/features.pt"
            print(f"\n💾 Saving features to {features_path}")
            print(f"  - Vproj shape: {features['Vproj_vecs'].shape}")
            print(f"  - T_fused shape: {features['T_fused_vecs'].shape}")
            
            with fs.open(features_path, 'wb') as f:
                torch.save(features_dict, f)
            print(f"✓ Features saved")
        else:
            print(f"\n⚠️ No features to save (features={features is not None})")
            if features:
                print(f"   Vproj_vecs present: {features.get('Vproj_vecs') is not None}")
        
        # 3. Save metrics
        metrics_with_meta = {
            'experiment_name': experiment_name,
            'dataset_name': dataset_name,
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        metrics_path = f"gs://{output_dir}/metrics.json"
        print(f"\n💾 Saving metrics to {metrics_path}")
        with fs.open(metrics_path, 'w') as f:
            json.dump(metrics_with_meta, f, indent=2)
        print(f"✓ Metrics saved")
        
        # 4. Create index entry
        index_entry = {
            'experiment_name': experiment_name,
            'dataset_name': dataset_name,
            'timestamp': timestamp,
            'output_dir': output_dir,
            'num_samples': len(results),
            'metrics': metrics,
            'files': {
                'predictions': predictions_path,
                'features': features_path if (features and features.get('Vproj_vecs') is not None) else None,
                'metrics': metrics_path
            }
        }
        
        index_path = f"gs://{config.GCS_BUCKET}/{config.OUTPUT_DIR}/evaluation_index.jsonl"
        print(f"\n📑 Updating index: {index_path}")
        try:
            with fs.open(index_path, 'a') as f:
                f.write(json.dumps(index_entry) + '\n')
            print(f"✓ Index updated")
        except:
            with fs.open(index_path, 'w') as f:
                f.write(json.dumps(index_entry) + '\n')
            print(f"✓ Index created")
        
        print(f"\n{'='*80}")
        print(f"✅ All files saved successfully!")
        print(f"{'='*80}")
        
        return output_dir, index_entry
        
    except Exception as e:
        print(f"\n❌ Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ================================
# MAIN
# ================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to LoRA checkpoint (GCS path)")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Experiment name (e.g., 'llm_only')")
    parser.add_argument("--dataset_name", type=str, default="vqav2_test",
                       help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit samples (for testing)")
    
    args = parser.parse_args()
    
    config = TestConfig()
    config.BATCH_SIZE = args.batch_size
    if args.max_samples:
        config.MAX_SAMPLES = args.max_samples
    
    print("\n" + "="*80)
    print("VLM EVALUATION - TEST RUN")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Test samples: {config.MAX_SAMPLES or 'ALL'}")
    print("="*80)
    
    # Load model
    model, processor = load_model_and_processor(config, args.checkpoint)
    
    # Load dataset
    dataset = load_test_dataset(config)
    
    # Run evaluation
    results, features = run_evaluation(model, processor, dataset, config)
    
    # Compute metrics
    metrics = compute_metrics(results, features)
    
    # Save results (NOW WITH ALL REQUIRED ARGS)
    output_dir, index_entry = save_results(
        results=results,
        features=features,
        metrics=metrics,
        config=config,
        experiment_name=args.experiment_name,  # ✅ Added
        dataset_name=args.dataset_name         # ✅ Added
    )
    
    # Summary
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if 'modality_gap' in metrics:
        print(f"Modality Gap: {metrics['modality_gap']:.4f}")
    if 'cka_img_txt' in metrics:
        print(f"CKA (img-txt): {metrics['cka_img_txt']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()