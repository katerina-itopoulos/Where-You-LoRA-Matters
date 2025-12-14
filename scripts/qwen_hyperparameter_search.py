import sys
sys.path.insert(0, '/home/sagemaker-user/Where-You-LoRA-Matters')
import os
os.environ['HF_DATASETS_CACHE'] = '/mnt/sagemaker-nvme/hf_cache'
import wandb
import itertools
import json
from datetime import datetime
import torch
from datasets import Dataset
from transformers import AutoProcessor
from src.finetuning_utils import (
    create_lora_config_vl,
    setup_optimizer_scheduler,
    train_vl_lora_with_wandb,
)
from src.model_utils import setup_vl_model_and_processor
from src.validation_utils import generate_validation_configs, create_validation_run_name
from datasets import Dataset, load_dataset
wandb.login()

GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/preprocessed_vqa"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MIN_PIXELS = 128*32*32  # 131,072 pixels (Qwen3-VL uses 32x32 patches!)
MAX_PIXELS = 256*32*32  # 262,144 pixels

# Hyperparameter search space
LORA_RANKS = [32, 64]
LEARNING_RATES = [1e-4, 2e-4, 5e-4]

# Fixed LoRA settings
LORA_ALPHA_MULTIPLIER = 2
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLM attention only

TRAIN_SIZE = 2000
VAL_SIZE = 200
TEST_SIZE = 200

# Training configuration
EPOCHS = 2
MAX_STEPS = -1
EVAL_STRATEGY = "epoch" 
SAVE_STRATEGY = "epoch"  
LOGGING_STEPS = 100

# Batch configuration
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4

# Early stopping
EARLY_STOPPING_PATIENCE = 3  # liStop if no improvement for 3 epochs
EARLY_STOPPING_THRESHOLD = 0.01

# Weights & Biases
WANDB_PROJECT = "qwen3-hyperparam-search"
WANDB_ENTITY = None

if __name__ == "__main__":
    
    validation_configs = list(generate_validation_configs(
    lora_ranks=LORA_RANKS,
    learning_rates=LEARNING_RATES,
    lora_alpha_multiplier=LORA_ALPHA_MULTIPLIER))

    total_runs = len(validation_configs)

    print("="*70)
    print(f"HYPERPARAMETER VALIDATION: {total_runs} configurations")
    print("="*70)
    print(f"Strategy: LLM-only LoRA (attention layers)")
    print(f"Ranks: {LORA_RANKS}")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Training samples: {TRAIN_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("="*70)

    print("\n" + "="*70)
    print("LOADING PREPROCESSED DATASETS FROM GCS")
    print("="*70)

    # Load processor (just for collator, not for preprocessing)
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )

    print("Loading train dataset...")

    train_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}_6k/train/train_shard_*.parquet",
        split="train"
    ).select(range(TRAIN_SIZE))

    print("Loading val dataset...")
    val_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}_6k/val_loss/val_loss_shard_*.parquet",
        split="train"
    ).select(range(VAL_SIZE))

    print("Loading test dataset...")
    test_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}_6k/test/test_shard_*.parquet",
        split="train"
    ).select(range(TEST_SIZE))

    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")

    cols = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    cols = [c for c in cols if c in train_dataset.column_names]

    train_dataset.set_format(type="torch", columns=cols)
    val_dataset.set_format(type="torch", columns=cols)
    test_dataset.set_format(type="torch", columns=cols)

    
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
        print(f"Target modules: {TARGET_MODULES}")

        try:
            lora_config = create_lora_config_vl(
                lora_r=rank,
                lora_alpha=alpha,
                lora_dropout=LORA_DROPOUT,
                target_modules=TARGET_MODULES
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
                warmup_steps=10,
                weight_decay=0.01,
            )

            wandb_config = {
                "experiment_type": "hyperparameter_validation",
                "placement_strategy": "llm_only",
                "model_name": MODEL_NAME,
                "lora_r": rank,
                "lora_alpha": alpha,
                "lora_alpha_multiplier": LORA_ALPHA_MULTIPLIER,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": TARGET_MODULES,
                "num_target_modules": len(TARGET_MODULES),
                "learning_rate": lr,
                "lr_scheduler_type": "cosine",
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
                "train_size": TRAIN_SIZE,
                "val_size": VAL_SIZE,
                "test_size": TEST_SIZE,
                "epochs": EPOCHS,
                "trainable_params": trainable_params,
                "trainable_percentage": 100*trainable_params/total_params,
            }

            run_name = create_validation_run_name(rank, lr, run_id)
            output_dir = f"./validation_outputs/{run_name}"

            print(f"\nStarting training: {run_name}")

            trainer = train_vl_lora_with_wandb(
                # Required
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,

                # W&B
                wandb_project=WANDB_PROJECT,
                wandb_run_name=run_name,
                wandb_entity=WANDB_ENTITY,
                wandb_config=wandb_config,

                # Output
                output_dir=output_dir,

                # Training schedule
                epochs=EPOCHS,
                max_steps=-1,
                # Remove eval_steps and save_steps - not needed for epoch-based!
                logging_steps=LOGGING_STEPS,

                # Batch size
                batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM_STEPS,

                # Optimizer
                optimizer_config=optimizer_config,

                # Early stopping
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
            )

            # Get best validation loss
            best_val_loss = min([log.get("eval_loss", float('inf'))
                                for log in trainer.state.log_history
                                if "eval_loss" in log])

            # Log success
            results.append({
                "run_id": run_id,
                "run_name": run_name,
                "status": "success",
                "lora_r": rank,
                "learning_rate": lr,
                "best_val_loss": best_val_loss,
                "trainable_params": trainable_params,
            })

            print(f"✓ Run {run_id} completed successfully")
            print(f"✓ Best val loss: {best_val_loss:.4f}")

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

    print(f"\n✓ Check W&B dashboard: wandb.ai")
    print(f"✓ Total runs: {len(results)} ({len(successful_runs)} successful)")
    print("="*70)