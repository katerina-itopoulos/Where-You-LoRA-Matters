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
    create_optimizer_with_param_groups,
)
from src.model_utils import setup_vl_model_and_processor
from datasets import Dataset, load_dataset
from datetime import datetime

PLACEMENT_STRATEGY = "vision_llm" 
EXPERIMENT_TYPE = "qwen3vl_vqa_vision_llm_lora_finetune" 

#Constants
GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/preprocessed_vqa_lowres_with_instruction"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

RANDOM_SEED = 42
MIN_PIXELS = 196*32*32
MAX_PIXELS = 196*32*32

WEIGHT_DECAY = 0.01
WARM_UP_STEPS = 500

VISION_RANK = 64
LLM_RANK = 64
VISION_LR = 1e-5  # Lower for stability
LLM_LR = 1e-4 

LORA_DROPOUT = 0.05
LORA_ALPHA_MULTIPLIER = 2

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

TARGET_VISION_LLM = TARGET_MODULES_VISION + TARGET_MODULES_LLM

TRAIN_SIZE = 20000
VAL_LOSS_SIZE = 1000
VAL_ACCURACY_SIZE = 1000
TEST_SIZE = 2000

# Training configuration
EPOCHS = 1.2  # ~3000 steps
MAX_STEPS = -1

EVAL_STRATEGY = "steps"
EVAL_STEPS = 500

SAVE_STRATEGY = "steps"  
SAVE_STEPS = 500 
LOGGING_STEPS = 250

# Checkpoint saving at specific steps
SAVE_CHECKPOINTS_AT_STEPS = [0, 500, 1000, 1500, 2000, 2500, 3000]

# Instruction already in data
ADD_SHORT_ANSWER_INSTRUCTION = False

# Batch configuration - SAME AS OTHER RUNS
BATCH_SIZE = 2  # Keep same as other configs
GRAD_ACCUM_STEPS = 4  # Keep same (effective batch = 8)
MAX_GRAD_NORM = 1.0

# Early stopping
EARLY_STOPPING_PATIENCE = None
EARLY_STOPPING_THRESHOLD = None

# VQA Accuracy Evaluation
COMPUTE_VQA_ACCURACY = True
VQA_EVAL_SAMPLES = 1000
VQA_BATCH_SIZE = 2

# Weights & Biases
WANDB_PROJECT = "qwen3vl-lora-experiments-vqa"
WANDB_ENTITY = None

LOG_DIR = "./logs"

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

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

if __name__ == "__main__":
    
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(
        LOG_DIR,
        f"vision_llm_qwen3vl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file = open(log_filename, "w")
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"[LOGGING] All prints will also go to: {log_filename}")
    
    set_random_seed(RANDOM_SEED)

    print("="*70)
    print(f"VISION + LLM FINE-TUNING (NO GRADIENT CHECKPOINTING)")
    print("="*70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Strategy: VISION + LLM")
    print(f"Vision: rank {VISION_RANK}, lr {VISION_LR}")
    print(f"LLM: rank {LLM_RANK}, lr {LLM_LR}")
    print(f"Training samples: {TRAIN_SIZE}")
    print(f"Epochs: {EPOCHS} (~3000 steps)")
    print(f"Batch size: {BATCH_SIZE} (same as other configs)")
    print(f"Grad accum: {GRAD_ACCUM_STEPS} (effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"Checkpoint steps: {SAVE_CHECKPOINTS_AT_STEPS}")
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

    print("Loading val_loss dataset...")
    val_loss_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}/val_loss/val_loss_shard_*.parquet",
        split="train"
    ).select(range(VAL_LOSS_SIZE))

    print("Loading valid dataset...")
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
    print("SETTING UP MODEL - VISION + LLM")
    print("="*70)

    # Create LoRA config with rank_pattern
    lora_config = create_lora_config_vl_with_rank_pattern(
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_VISION_LLM,
        rank_pattern={
            r".*visual\.blocks.*": VISION_RANK,
            r".*language_model.*": LLM_RANK,
        },
        alpha_pattern={
            r".*visual\.blocks.*": VISION_RANK * LORA_ALPHA_MULTIPLIER,
            r".*language_model.*": LLM_RANK * LORA_ALPHA_MULTIPLIER,
        }
    )

    model, _, trainable_params, total_params = setup_vl_model_and_processor(
        model_name=MODEL_NAME,
        lora_config=lora_config,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS
    )
    
    model.config.use_cache = False
    
    # CRITICAL: Disable gradient checkpointing to allow vision gradients!
    print("\n" + "="*70)
    print("GRADIENT CHECKPOINTING DISABLED")
    print("="*70)
    print("This is REQUIRED for vision gradients to flow properly!")
    print("Memory usage will be higher, but vision LoRA will actually train.")
    print("="*70 + "\n")
    # model.gradient_checkpointing_enable()  # <-- DISABLED!

    print(f"✓ Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Debug: check trainable parameters
    print("\n=== DEBUG: Trainable Parameter Names (first few) ===")
    vision_count = 0
    proj_count = 0
    llm_count = 0
    other_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if vision_count < 3 or llm_count < 3:
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
    print(f"  Vision: {vision_count} (should be >0)")
    print(f"  Projector: {proj_count} (should be 0!)")
    print(f"  LLM: {llm_count} (should be >0)")
    print(f"  Other (unmatched): {other_count}")
    print("="*50)

    # Setup optimizer with different LRs
    print("\n" + "="*70)
    print("CREATING OPTIMIZER WITH SEPARATE LRS")
    print("="*70)
    
    optimizer = create_optimizer_with_param_groups(
        model=model,
        vision_lr=VISION_LR,
        llm_lr=LLM_LR,
        projector_lr=0,  # Not used
        weight_decay=WEIGHT_DECAY,
    )
    
    wandb_config = {
        "experiment_type": EXPERIMENT_TYPE,
        "placement_strategy": PLACEMENT_STRATEGY,
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "vision_rank": VISION_RANK,
        "llm_rank": LLM_RANK,
        "vision_lr": VISION_LR,
        "llm_lr": LLM_LR,
        "lora_alpha_multiplier": LORA_ALPHA_MULTIPLIER,
        "lora_dropout": LORA_DROPOUT,
        "target_modules_vision": TARGET_MODULES_VISION,
        "target_modules_llm": TARGET_MODULES_LLM,
        "num_target_modules": len(TARGET_VISION_LLM),
        "lr_scheduler_type": "cosine",
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "train_size": TRAIN_SIZE,
        "val_loss_size": VAL_LOSS_SIZE,
        "val_accuracy_size": VAL_ACCURACY_SIZE,
        "test_size": TEST_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "trainable_params": trainable_params,
        "trainable_percentage": 100*trainable_params/total_params,
        "checkpoint_steps": SAVE_CHECKPOINTS_AT_STEPS,
        "gradient_checkpointing": False,
    }

    run_name = f"vision_llm_vr{VISION_RANK}_lr{LLM_RANK}_vlr{VISION_LR}_llr{LLM_LR}_no_grad_ckpt"
    output_dir = f"./validation_outputs/{run_name}"

    print(f"\nStarting training: {run_name}")

    try:
        # Note: You'll need to modify train_vl_lora_with_wandb to accept
        # a pre-built optimizer if using separate LRs
        # For now, using simple single LR approach
        
        optimizer_config = setup_optimizer_scheduler(
            learning_rate=VISION_LR,  # Use lower LR for both
            lr_scheduler_type="cosine",
            warmup_steps=WARM_UP_STEPS,
            weight_decay=WEIGHT_DECAY,
        )
        
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
            
            # Checkpoint saving
            save_checkpoints_at_steps=SAVE_CHECKPOINTS_AT_STEPS,
            add_short_answer_instruction=ADD_SHORT_ANSWER_INSTRUCTION,

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

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"✓ Best val loss: {run_results['best_val_loss']:.4f}")
        
        if COMPUTE_VQA_ACCURACY:
            print(f"✓ Best VQA accuracy: {run_results['best_vqa_accuracy']:.2%}")
            print(f"✓ Best exact match: {run_results['best_exact_match']:.2%}")
        
        print(f"✓ Output saved to: {output_dir}")
        print(f"✓ Checkpoints saved at steps: {SAVE_CHECKPOINTS_AT_STEPS}")
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