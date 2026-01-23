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
    create_lora_config_vl_with_rank_pattern,
    setup_optimizer_scheduler,
    train_vl_lora_with_wandb,
)
from src.model_utils import setup_vl_model_and_processor
from datasets import Dataset, load_dataset

PLACEMENT_STRATEGY = "vision_proj"
EXPERIMENT_TYPE = "qwen3vl_vqa_vision_proj_lora_finetune"

#Constants
GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/preprocessed_vqa_lowres"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

RANDOM_SEED = 42
MIN_PIXELS = 196*32*32
MAX_PIXELS = 196*32*32

WEIGHT_DECAY = 0.01
WARM_UP_STEPS = 500

VISION_RANK = 64
PROJECTOR_RANK = 64
VISION_LR = 1e-4
PROJECTOR_LR = 1e-4

LORA_DROPOUT = 0.05

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

TRAIN_SIZE = 10000
VAL_LOSS_SIZE = 500
VAL_ACCURACY_SIZE = 500
TEST_SIZE = 500

# Training configuration
EPOCHS = 2
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
VQA_EVAL_SAMPLES = 500

# Weights & Biases
WANDB_PROJECT = "qwen3vl-experiments-vqa-projvision-test"
WANDB_ENTITY = None

def set_random_seed(seed):
    """
    Set random seed for reproducibility across PyTorch, NumPy, and Python.
    """
    print(f"\n{'='*70}")
    print(f"🎲 Setting Random Seed: {seed}")
    print(f"{'='*70}\n")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

if __name__ == "__main__":
    
    set_random_seed(RANDOM_SEED)

    print("="*70)
    print(f"VISION + PROJECTOR TRIAL RUN")
    print("="*70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Strategy: VISION + PROJECTOR (separate ranks & LRs)")
    print(f"Vision: rank {VISION_RANK}, lr {VISION_LR}")
    print(f"Projector: rank {PROJECTOR_RANK}, lr {PROJECTOR_LR}")
    print(f"Training samples: {TRAIN_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"VQA Accuracy: {'Enabled' if COMPUTE_VQA_ACCURACY else 'Disabled'}")
    print("="*70)

    print("\n" + "="*70)
    print("LOADING PREPROCESSED DATASETS FROM GCS")
    print("="*70)

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

    # Set format for datasets
    train_cols = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    train_cols = [c for c in train_cols if c in train_dataset.column_names]
    train_dataset.set_format(type="torch", columns=train_cols)
    
    val_loss_cols = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    val_loss_cols = [c for c in val_loss_cols if c in val_loss_dataset.column_names]
    val_loss_dataset.set_format(type="torch", columns=val_loss_cols)
    
    val_accuracy_cols = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "answer", "answer_counts"]
    val_accuracy_cols = [c for c in val_accuracy_cols if c in val_accuracy_dataset.column_names]
    val_accuracy_dataset.set_format(type="torch", columns=val_accuracy_cols)
    
    test_cols = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "answer", "answer_counts"]
    test_cols = [c for c in test_cols if c in test_dataset.column_names]
    test_dataset.set_format(type="torch", columns=test_cols)

    print("\n" + "="*70)
    print("SETTING UP MODEL WITH SEPARATE RANKS FOR VISION AND PROJECTOR")
    print("="*70)

    # Create LoRA config with rank_pattern using REGEX patterns
    lora_config = create_lora_config_vl_with_rank_pattern(
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_VISION_PROJECTOR,
        rank_pattern={
            r".*visual\.blocks.*": VISION_RANK,        # Vision encoder blocks
            r".*visual\.merger.*": PROJECTOR_RANK,     # Merger projector
            r".*visual\.deepstack.*": PROJECTOR_RANK,  # Deepstack projectors
        },
        alpha_pattern={
            r".*visual\.blocks.*": VISION_RANK * 2,
            r".*visual\.merger.*": PROJECTOR_RANK * 2,
            r".*visual\.deepstack.*": PROJECTOR_RANK * 2,
        }
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

    # Debug: check trainable parameter names and matching
    print("\n=== DEBUG: Trainable Parameter Names ===")
    vision_count = 0
    proj_count = 0
    llm_count = 0
    other_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
            if "visual.blocks" in name:
                vision_count += 1
            elif "visual.merger" in name or "visual.deepstack" in name:
                proj_count += 1
            elif "language_model" in name:
                llm_count += 1
            else:
                other_count += 1

    print(f"\nMatched counts:")
    print(f"  Vision: {vision_count}")
    print(f"  Projector: {proj_count}")
    print(f"  LLM: {llm_count}")
    print(f"  Other (unmatched): {other_count}")
    print("="*50)

    # Verify rank assignment - check actual rank values
    # Verify rank by checking lora_A weight shapes
    print("\n=== Verifying LoRA Rank Assignment ===")
    vision_ranks = []
    proj_ranks = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name:  # lora_A shape is [rank, in_features]
            rank = param.shape[0]
            if 'visual.blocks' in name:
                vision_ranks.append(rank)
                if len(vision_ranks) <= 3:
                    print(f"Vision: {name}: rank={rank}")
            elif 'visual.merger' in name or 'visual.deepstack' in name:
                proj_ranks.append(rank)
                if len(proj_ranks) <= 3:
                    print(f"Projector: {name}: rank={rank}")

    print(f"\nVision LoRA modules: {len(vision_ranks)}, ranks: {set(vision_ranks)}")
    print(f"Projector LoRA modules: {len(proj_ranks)}, ranks: {set(proj_ranks)}")
    print("="*70)

    # Setup optimizer config
    optimizer_config = setup_optimizer_scheduler(
        learning_rate=VISION_LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARM_UP_STEPS,
        weight_decay=WEIGHT_DECAY,
    )

    wandb_config = {
        "experiment_type": EXPERIMENT_TYPE,
        "placement_strategy": PLACEMENT_STRATEGY,
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "vision_rank": VISION_RANK,
        "projector_rank": PROJECTOR_RANK,
        "vision_lr": VISION_LR,
        "projector_lr": PROJECTOR_LR,
        "lora_dropout": LORA_DROPOUT,
        "target_modules_vision": TARGET_MODULES_VISION,
        "target_modules_projector": TARGET_MODULES_PROJECTOR,
        "num_target_modules": len(TARGET_VISION_PROJECTOR),
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

    run_name = f"vision_proj_trial_r{VISION_RANK}_{PROJECTOR_RANK}_lr{VISION_LR}_{PROJECTOR_LR}"
    output_dir = f"./validation_outputs/{run_name}"

    print(f"\nStarting training: {run_name}")

    try:
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

            # Optimizer with separate LRs
            optimizer_config=optimizer_config,
            use_separate_lrs=True,
            vision_lr=VISION_LR,
            projector_lr=PROJECTOR_LR,

            # Early stopping
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
           
            # VQA Accuracy
            compute_vqa_accuracy=COMPUTE_VQA_ACCURACY,
            vqa_eval_samples=VQA_EVAL_SAMPLES,
        )

        print("\n" + "="*70)
        print("TRIAL RUN COMPLETE!")
        print("="*70)
        print(f"✓ Best val loss: {run_results['best_val_loss']:.4f}")
        
        if COMPUTE_VQA_ACCURACY:
            print(f"✓ Best VQA accuracy: {run_results['best_vqa_accuracy']:.2%}")
            print(f"✓ Best exact match: {run_results['best_exact_match']:.2%}")
        
        print(f"✓ Output saved to: {output_dir}")
        print(f"✓ Check W&B dashboard: wandb.ai")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        torch.cuda.empty_cache()
        import gc
        gc.collect()