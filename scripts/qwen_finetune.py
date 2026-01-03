#torch version needs to be 2.8.0
import sys
import os
import wandb
import torch
import numpy as np
import random
sys.path.insert(0, '/home/sagemaker-user/Where-You-LoRA-Matters')
os.environ['HF_DATASETS_CACHE'] = '/mnt/sagemaker-nvme/hf_cache'
wandb.login()

from transformers import AutoProcessor, set_seed
from src.finetuning_utils import (
    create_lora_config_vl,
    setup_optimizer_scheduler,
    train_vl_lora_with_wandb,
)
from src.model_utils import setup_vl_model_and_processor
from src.validation_utils import generate_validation_configs, create_validation_run_name
from datasets import Dataset, load_dataset

#torch.autograd.set_detect_anomaly(True)

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
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
#TARGET_MODULES_LLM = ["q_proj", "k_proj", "v_proj", "o_proj"]
#TARGET_MODULES_VISION = [r"visual\.blocks\.\d+\.attn\.qkv", r"visual\.blocks\.\d+\.attn\.proj"]
#TARGET_MODULES_PROJECTOR = [
#   r"visual\.merger\.linear_fc[12]",
#    r"visual\.deepstack_merger_list\.\d+\.linear_fc[12]",
#]

TARGET_VISION_PROJECTOR = TARGET_MODULES_VISION + TARGET_MODULES_PROJECTOR
TARGET_LLM_PROJECTOR = TARGET_MODULES_LLM + TARGET_MODULES_PROJECTOR

TRAIN_SIZE = 20000
VAL_LOSS_SIZE = 2000
VAL_ACCURACY_SIZE = 2000
TEST_SIZE = 5000

# Training configuration
EPOCHS = 2
MAX_STEPS = -1

EVAL_STRATEGY = "steps"
EVAL_STEPS = 250

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
    print(f"Strategy: LLM-only LoRA (attention layers)")
    print(f"Ranks: {LORA_RANKS}")
    print(f"Learning rates: {LEARNING_RATES}")
    print(f"Training samples: {TRAIN_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"VQA Accuracy: {'Enabled' if COMPUTE_VQA_ACCURACY else 'Disabled'}")
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
                warmup_steps=WARM_UP_STEPS,
                weight_decay=WEIGHT_DECAY,
            )

            wandb_config = {
                "experiment_type": "hyperparameter_validation",
                "placement_strategy": "llm_only",
                "model_name": MODEL_NAME,
                "random_seed": RANDOM_SEED,
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
                "val_loss_size": VAL_LOSS_SIZE,
                "val_accuracy_size": VAL_ACCURACY_SIZE,
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
                processor=processor,
                train_dataset=train_dataset,
                val_dataset=val_loss_dataset,  # ← Use val_loss for computing loss!
                val_accuracy_dataset=val_accuracy_dataset,  # ← Pass accuracy dataset separately!
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
            )

            # Get best validation loss
            best_val_loss = min([log.get("eval_loss", float('inf'))
                                for log in trainer.state.log_history
                                if "eval_loss" in log])

            # Log success
            result_dict = {
                "run_id": run_id,
                "run_name": run_name,
                "status": "success",
                "lora_r": rank,
                "learning_rate": lr,
                "best_val_loss": best_val_loss,
                "trainable_params": trainable_params,
            }
            
            # Add VQA accuracy if it was computed
            if COMPUTE_VQA_ACCURACY:
                result_dict["best_vqa_accuracy"] = wandb.run.summary.get("best_vqa_accuracy", None)
                result_dict["best_exact_match_accuracy"] = wandb.run.summary.get("best_exact_match_accuracy", None)
            
            results.append(result_dict)

            print(f"✓ Run {run_id} completed successfully")
            print(f"✓ Best val loss: {best_val_loss:.4f}")
            if COMPUTE_VQA_ACCURACY:
                print(f"✓ Best VQA accuracy: {result_dict.get('best_vqa_accuracy', 0):.2%}")
                print(f"✓ Best exact match: {result_dict.get('best_exact_match_accuracy', 0):.2%}")

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