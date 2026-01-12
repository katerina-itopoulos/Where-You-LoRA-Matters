#torch version needs to be 2.8.0
import sys
import os
import wandb
import torch
import numpy as np
import random
sys.path.insert(0, '/home/ubuntu/Where-You-LoRA-Matters')
os.environ['HF_DATASETS_CACHE'] = '/opt/dlami/nvme/hf_cache'
wandb.login()
import regex as re
from transformers import AutoProcessor, set_seed
from src.finetuning_utils import (
    create_lora_config_vl,
    create_lora_config_vl_with_rank_pattern,
    setup_optimizer_scheduler,
    train_vl_lora_with_wandb,
)
from src.model_utils import setup_vl_model_and_processor
from src.validation_utils import generate_validation_configs, create_validation_run_name
from datasets import Dataset, load_dataset

#torch.autograd.set_detect_anomaly(True)

PLACEMENT_STRATEGY = "llm_only"
EXPERIMENT_TYPE = "qwen3vl_vqa_llm_lora_finetune"

#Constants
GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/preprocessed_vqa_lowres"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

RANDOM_SEED = 42
MIN_PIXELS = 196*32*32
MAX_PIXELS = 196*32*32

WEIGHT_DECAY = 0.01
WARM_UP_STEPS = 500

# Hyperparameter search space
LORA_RANKS = [32]
LEARNING_RATES = [5e-5]

# Fixed LoRA settings
LORA_ALPHA_MULTIPLIER = 2
LORA_DROPOUT = 0.05

#TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
TARGET_MODULES_LLM = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Vision encoder layers (27 blocks)
TARGET_MODULES_VISION = [
    "visual.blocks.0.attn.qkv", "visual.blocks.0.attn.proj",
    "visual.blocks.1.attn.qkv", "visual.blocks.1.attn.proj",
    "visual.blocks.2.attn.qkv", "visual.blocks.2.attn.proj",
    "visual.blocks.3.attn.qkv", "visual.blocks.3.attn.proj",
    "visual.blocks.4.attn.qkv", "visual.blocks.4.attn.proj",
    "visual.blocks.5.attn.qkv", "visual.blocks.5.attn.proj",
    "visual.blocks.6.attn.qkv", "visual.blocks.6.attn.proj",
    "visual.blocks.7.attn.qkv", "visual.blocks.7.attn.proj",
    "visual.blocks.8.attn.qkv", "visual.blocks.8.attn.proj",
    "visual.blocks.9.attn.qkv", "visual.blocks.9.attn.proj",
    "visual.blocks.10.attn.qkv", "visual.blocks.10.attn.proj",
    "visual.blocks.11.attn.qkv", "visual.blocks.11.attn.proj",
    "visual.blocks.12.attn.qkv", "visual.blocks.12.attn.proj",
    "visual.blocks.13.attn.qkv", "visual.blocks.13.attn.proj",
    "visual.blocks.14.attn.qkv", "visual.blocks.14.attn.proj",
    "visual.blocks.15.attn.qkv", "visual.blocks.15.attn.proj",
    "visual.blocks.16.attn.qkv", "visual.blocks.16.attn.proj",
    "visual.blocks.17.attn.qkv", "visual.blocks.17.attn.proj",
    "visual.blocks.18.attn.qkv", "visual.blocks.18.attn.proj",
    "visual.blocks.19.attn.qkv", "visual.blocks.19.attn.proj",
    "visual.blocks.20.attn.qkv", "visual.blocks.20.attn.proj",
    "visual.blocks.21.attn.qkv", "visual.blocks.21.attn.proj",
    "visual.blocks.22.attn.qkv", "visual.blocks.22.attn.proj",
    "visual.blocks.23.attn.qkv", "visual.blocks.23.attn.proj",
    "visual.blocks.24.attn.qkv", "visual.blocks.24.attn.proj",
    "visual.blocks.25.attn.qkv", "visual.blocks.25.attn.proj",
    "visual.blocks.26.attn.qkv", "visual.blocks.26.attn.proj",
]

# Projector layers (merger + deepstack mergers)
TARGET_MODULES_PROJECTOR = [
    "visual.merger.linear_fc1", "visual.merger.linear_fc2",
    "visual.deepstack_merger_list.0.linear_fc1", "visual.deepstack_merger_list.0.linear_fc2",
    "visual.deepstack_merger_list.1.linear_fc1", "visual.deepstack_merger_list.1.linear_fc2",
    "visual.deepstack_merger_list.2.linear_fc1", "visual.deepstack_merger_list.2.linear_fc2",
]

TARGET_VISION_PROJECTOR = TARGET_MODULES_VISION + TARGET_MODULES_PROJECTOR
TARGET_LLM_PROJECTOR = TARGET_MODULES_LLM + TARGET_MODULES_PROJECTOR

TRAIN_SIZE = 20000
VAL_LOSS_SIZE = 2000
VAL_ACCURACY_SIZE = 2000
TEST_SIZE = 5000

# Training configuration
EPOCHS = 2.4
MAX_STEPS = -1

EVAL_STRATEGY = "steps"
EVAL_STEPS = 500

SAVE_STRATEGY = "steps"  
SAVE_STEPS = 500 
LOGGING_STEPS = 250

# Batch configuration
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_GRAD_NORM = 1.0

# Early stopping
EARLY_STOPPING_PATIENCE = None
EARLY_STOPPING_THRESHOLD = None

# VQA Accuracy Evaluation
COMPUTE_VQA_ACCURACY = True
VQA_EVAL_SAMPLES = 2000
VQA_BATCH_SIZE = 4 

# Weights & Biases
WANDB_PROJECT = "qwen3vl-experiments-vqa"
WANDB_ENTITY = None

def set_random_seed(seed):
    """
    Set random seed for reproducibility across PyTorch, NumPy, and Python.
    """
    print(f"\n{'='*70}")
    print(f"🎲 Setting Random Seed: {seed}")
    print(f"{'='*70}\n")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # HuggingFace transformers
    set_seed(seed)

if __name__ == "__main__":
    
    # Set random seed FIRST!
    set_random_seed(RANDOM_SEED)
    
    validation_configs = list(generate_validation_configs(
        lora_ranks=LORA_RANKS,
        learning_rates=LEARNING_RATES,
        lora_alpha_multiplier=LORA_ALPHA_MULTIPLIER
    ))

    total_runs = len(validation_configs)

    print("="*70)
    print(f"HYPERPARAMETER VALIDATION: {total_runs} configurations")
    print("="*70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Strategy: LLM ONLY")
    print(f"Ranks: {LORA_RANKS}")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Training samples: {TRAIN_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"VQA Accuracy: {'Enabled' if COMPUTE_VQA_ACCURACY else 'Disabled'}")
    print(f"VQA Batch Size: {VQA_BATCH_SIZE}")
    print("="*70)

    print("\n" + "="*70)
    print("LOADING PREPROCESSED DATASETS FROM GCS")
    print("="*70)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )

    print("Loading train dataset...")
    train_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}/train/train_shard_*.parquet",
        split="train"
    ).select(range(TRAIN_SIZE))

    print("Loading val_loss dataset (for computing validation loss)...")
    val_loss_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}/val_loss/val_loss_shard_*.parquet",
        split="train"
    ).select(range(VAL_LOSS_SIZE))

    print("Loading valid dataset (for computing VQA accuracy)...")
    val_accuracy_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}/valid/valid_shard_*.parquet",
        split="train"
    ).select(range(VAL_ACCURACY_SIZE))

    print("Loading test dataset...")
    test_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}/test/test_shard_*.parquet",
        split="train"
    ).select(range(TEST_SIZE))

    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val (loss): {len(val_loss_dataset)} samples")
    print(f"✓ Val (accuracy): {len(val_accuracy_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")

    # Set format for training dataset
    train_cols = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    train_cols = [c for c in train_cols if c in train_dataset.column_names]
    train_dataset.set_format(type="torch", columns=train_cols)
    
    # Set format for val_loss dataset (same as training - has labels)
    val_loss_cols = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    val_loss_cols = [c for c in val_loss_cols if c in val_loss_dataset.column_names]
    val_loss_dataset.set_format(type="torch", columns=val_loss_cols)
    
    # Set format for val_accuracy dataset (for VQA metric - has answer_counts)
    val_accuracy_cols = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "answer", "answer_counts"]
    val_accuracy_cols = [c for c in val_accuracy_cols if c in val_accuracy_dataset.column_names]
    val_accuracy_dataset.set_format(type="torch", columns=val_accuracy_cols)
    
    # Set format for test dataset
    test_cols = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "answer", "answer_counts"]
    test_cols = [c for c in test_cols if c in test_dataset.column_names]
    test_dataset.set_format(type="torch", columns=test_cols)

    results = []

    for run_id, config in enumerate(validation_configs, 1):
        rank = config["lora_r"]
        lr = config["learning_rate"]
        alpha = config["lora_alpha"]

        print("\n" + "="*70)
        print(f"VALIDATION RUN {run_id}/{total_runs}")
        print("="*70)
        print(f"LoRA rank: {rank}")
        print(f"LoRA alpha: {alpha} ({LORA_ALPHA_MULTIPLIER}x)")
        print(f"Learning rate: {lr}")
        print(f"Target modules: {TARGET_MODULES_LLM}")

        try:
            lora_config = create_lora_config_vl(
                lora_r=rank,
                lora_alpha=alpha,
                lora_dropout=LORA_DROPOUT,
                target_modules=TARGET_MODULES_LLM
            )

            model, _, trainable_params, total_params = setup_vl_model_and_processor(
                model_name=MODEL_NAME,
                lora_config=lora_config,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS
            )
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

            print(f"✓ Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

            optimizer_config = setup_optimizer_scheduler(
                learning_rate=lr,
                lr_scheduler_type="cosine",
                warmup_steps=WARM_UP_STEPS,
                weight_decay=WEIGHT_DECAY,
            )

            wandb_config = {
                "experiment_type": EXPERIMENT_TYPE,
                "placement_strategy": PLACEMENT_STRATEGY,
                "model_name": MODEL_NAME,
                "random_seed": RANDOM_SEED,
                "lora_r": rank,
                "lora_alpha": alpha,
                "lora_alpha_multiplier": LORA_ALPHA_MULTIPLIER,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": TARGET_MODULES_LLM,
                "num_target_modules": len(TARGET_MODULES_LLM),
                "learning_rate": lr,
                "lr_scheduler_type": "cosine",
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "train_size": TRAIN_SIZE,
                "val_loss_size": VAL_LOSS_SIZE,
                "val_accuracy_size": VAL_ACCURACY_SIZE,
                "test_size": TEST_SIZE,
                "epochs": EPOCHS,
                "trainable_params": trainable_params,
                "trainable_percentage": 100*trainable_params/total_params,
            }

            run_name = create_validation_run_name(rank, lr, run_id)
            output_dir = f"./validation_outputs_final_llm/{run_name}"

            print(f"\nStarting training: {run_name}")

            trainer, run_results = train_vl_lora_with_wandb(
                # Required
                model=model,
                processor=processor,
                train_dataset=train_dataset,
                val_dataset=val_loss_dataset,
                val_accuracy_dataset=val_accuracy_dataset,
                test_dataset=test_dataset,
                max_grad_norm=MAX_GRAD_NORM,

                # W&B
                wandb_project=WANDB_PROJECT,
                wandb_run_name=run_name,
                wandb_entity=WANDB_ENTITY,
                wandb_config=wandb_config,

                # Output
                output_dir=output_dir,

                # Training schedule
                epochs=EPOCHS,
                max_steps=MAX_STEPS,
                logging_steps=LOGGING_STEPS,
                eval_strategy=EVAL_STRATEGY,
                eval_steps=EVAL_STEPS,
                save_strategy=SAVE_STRATEGY,
                save_steps=SAVE_STEPS,

                # Batch size
                batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM_STEPS,

                # Optimizer
                optimizer_config=optimizer_config,

                # Early stopping
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
               
                # VQA Accuracy
                compute_vqa_accuracy=COMPUTE_VQA_ACCURACY,
                vqa_eval_samples=VQA_EVAL_SAMPLES,
                vqa_batch_size=VQA_BATCH_SIZE,
            )

            # Log success
            result_dict = {
                "run_id": run_id,
                "run_name": run_name,
                "status": "success",
                "lora_r": rank,
                "learning_rate": lr,
                "best_val_loss": run_results["best_val_loss"],
                "trainable_params": trainable_params,
            }
            
            # Add VQA accuracy if it was computed
            if COMPUTE_VQA_ACCURACY:
                result_dict["best_vqa_accuracy"] = run_results["best_vqa_accuracy"]
                result_dict["best_exact_match_accuracy"] = run_results["best_exact_match"]
            
            results.append(result_dict)

            print(f"✓ Run {run_id} completed successfully")
            print(f"✓ Best val loss: {run_results['best_val_loss']:.4f}")

            if COMPUTE_VQA_ACCURACY:
                print(f"✓ Best VQA accuracy: {run_results['best_vqa_accuracy']:.2%}")
                print(f"✓ Best exact match: {run_results['best_exact_match']:.2%}")

            # Clean up
            del model
            del trainer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ Run {run_id} failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

            results.append({
                "run_id": run_id,
                "run_name": create_validation_run_name(rank, lr, run_id),
                "status": "failed",
                "lora_r": rank,
                "learning_rate": lr,
                "error": str(e),
            })

            # Clean up even on failure
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            continue

    print("\n" + "="*70)
    print("VALIDATION COMPLETE - RESULTS ANALYSIS")
    print("="*70)

    successful_runs = [r for r in results if r["status"] == "success"]
    
    # Print summary with VQA accuracy
    if successful_runs and COMPUTE_VQA_ACCURACY:
        print("\nRESULTS SUMMARY:")
        print(f"{'Run':<30} {'Val Loss':<12} {'VQA Acc':<12} {'Exact Match':<12}")
        print("-" * 70)
        for r in successful_runs:
            vqa_acc = r.get('best_vqa_accuracy', 0) * 100 if r.get('best_vqa_accuracy') else 0
            exact_acc = r.get('best_exact_match_accuracy', 0) * 100 if r.get('best_exact_match_accuracy') else 0
            print(f"{r['run_name']:<30} {r['best_val_loss']:<12.4f} {vqa_acc:<12.2f}% {exact_acc:<12.2f}%")

    print(f"\n✓ Check W&B dashboard: wandb.ai")
    print(f"✓ Total runs: {len(results)} ({len(successful_runs)} successful)")
    print("="*70)