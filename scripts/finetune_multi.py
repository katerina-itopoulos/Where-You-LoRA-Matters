import gc
import random
import sys
import traceback

import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import AutoProcessor, set_seed

sys.path.insert(0, "/home/ubuntu/Where-You-LoRA-Matters")

from src.finetuning_utils import (
    create_lora_config_vl_with_rank_pattern,
    setup_optimizer_scheduler,
    train_vl_lora_with_wandb,
)
from src.model_utils import setup_vl_model_and_processor

PLACEMENT_STRATEGY = "vision_llm"
EXPERIMENT_TYPE = "qwen3vl_vqa_vision_llm_lora_finetune"

GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/preprocessed_vqa_lowres"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

RANDOM_SEED = 42
MIN_PIXELS = 196 * 32 * 32
MAX_PIXELS = 196 * 32 * 32

WEIGHT_DECAY = 0.01
WARM_UP_STEPS = 500

VISION_RANK = 64
LLM_RANK = 64
PROJ_RANK = 64
VISION_LR = 1e-5
PROJ_LR = 1e-4
LLM_LR = 1e-4

LORA_DROPOUT = 0.05

TARGET_MODULES_LLM = ["q_proj", "k_proj", "v_proj", "o_proj"]

TARGET_MODULES_VISION = [
    f"visual.blocks.{i}.attn.{op}"
    for i in range(27)
    for op in ("qkv", "proj")
]

TARGET_MODULES_PROJECTOR = [
    "visual.merger.linear_fc1", "visual.merger.linear_fc2",
    "visual.deepstack_merger_list.0.linear_fc1", "visual.deepstack_merger_list.0.linear_fc2",
    "visual.deepstack_merger_list.1.linear_fc1", "visual.deepstack_merger_list.1.linear_fc2",
    "visual.deepstack_merger_list.2.linear_fc1", "visual.deepstack_merger_list.2.linear_fc2",
]

TARGET_VISION_PROJECTOR = TARGET_MODULES_VISION + TARGET_MODULES_PROJECTOR
TARGET_LLM_PROJECTOR = TARGET_MODULES_LLM + TARGET_MODULES_PROJECTOR
TARGET_VISION_LLM = TARGET_MODULES_VISION + TARGET_MODULES_LLM

TRAIN_SIZE = 20000
VAL_LOSS_SIZE = 500
VAL_ACCURACY_SIZE = 500
TEST_SIZE = 500

EPOCHS = 2.4
MAX_STEPS = -1
EVAL_STRATEGY = "steps"
EVAL_STEPS = 500
SAVE_STRATEGY = "steps"
SAVE_STEPS = 500
LOGGING_STEPS = 250

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_GRAD_NORM = 1.0

EARLY_STOPPING_PATIENCE = None
EARLY_STOPPING_THRESHOLD = None

COMPUTE_VQA_ACCURACY = True
VQA_EVAL_SAMPLES = 500

WANDB_PROJECT = "qwen3vl-experiments-vqa"
WANDB_ENTITY = None


def set_random_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


def load_split(prefix: str, shard_glob: str, size: int):
    """Load a parquet split from GCS and select the first ``size`` rows."""
    return load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{prefix}/{shard_glob}",
        split="train",
    ).select(range(size))


def apply_format(dataset, columns: list[str]):
    """Set torch format on a dataset, filtering to only present columns."""
    cols = [c for c in columns if c in dataset.column_names]
    dataset.set_format(type="torch", columns=cols)
    return dataset


if __name__ == "__main__":
    wandb.login()
    set_random_seed(RANDOM_SEED)

    print("=" * 70)
    print("VISION + LLM LORA FINETUNING")
    print("=" * 70)
    print(f"Random Seed:       {RANDOM_SEED}")
    print(f"Strategy:          Vision + LLM")
    print(f"Vision:            rank {VISION_RANK}, lr {VISION_LR}")
    print(f"LLM:               rank {LLM_RANK}, lr {LLM_LR}")
    print(f"Training samples:  {TRAIN_SIZE}")
    print(f"Epochs:            {EPOCHS}")
    print(f"VQA Accuracy:      {'enabled' if COMPUTE_VQA_ACCURACY else 'disabled'}")
    print("=" * 70)

    print("\nLoading preprocessed datasets from GCS...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, trust_remote_code=True
    )

    train_dataset = apply_format(
        load_split("train", "train_shard_*.parquet", TRAIN_SIZE),
        ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"],
    )
    val_loss_dataset = apply_format(
        load_split("val_loss", "val_loss_shard_*.parquet", VAL_LOSS_SIZE),
        ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"],
    )
    val_accuracy_dataset = apply_format(
        load_split("valid", "valid_shard_*.parquet", VAL_ACCURACY_SIZE),
        ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "answer", "answer_counts"],
    )
    test_dataset = apply_format(
        load_split("test", "test_shard_*.parquet", TEST_SIZE),
        ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "answer", "answer_counts"],
    )

    print(f"Train: {len(train_dataset)}, Val loss: {len(val_loss_dataset)}, "
          f"Val accuracy: {len(val_accuracy_dataset)}, Test: {len(test_dataset)}")

    lora_config = create_lora_config_vl_with_rank_pattern(
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_VISION_LLM,
        rank_pattern={
            r".*visual\.blocks.*": VISION_RANK,
            r".*language_model.*": LLM_RANK,
        },
        alpha_pattern={
            r".*visual\.blocks.*": VISION_RANK * 2,
            r".*language_model.*": LLM_RANK * 2,
        },
    )

    model, _, trainable_params, total_params = setup_vl_model_and_processor(
        model_name=MODEL_NAME,
        lora_config=lora_config,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model.config.use_cache = False

    optimizer_config = setup_optimizer_scheduler(
        learning_rate=VISION_LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARM_UP_STEPS,
        weight_decay=WEIGHT_DECAY,
    )

    run_name = f"vision_llm_r{VISION_RANK}_{LLM_RANK}_lr{VISION_LR}_{LLM_LR}"

    wandb_config = {
        "experiment_type": EXPERIMENT_TYPE,
        "placement_strategy": PLACEMENT_STRATEGY,
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "vision_rank": VISION_RANK,
        "llm_rank": LLM_RANK,
        "vision_lr": VISION_LR,
        "llm_lr": LLM_LR,
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
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params,
    }

    print(f"\nStarting training: {run_name}")

    try:
        trainer, run_results = train_vl_lora_with_wandb(
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_loss_dataset,
            val_accuracy_dataset=val_accuracy_dataset,
            test_dataset=test_dataset,
            max_grad_norm=MAX_GRAD_NORM,
            wandb_project=WANDB_PROJECT,
            wandb_run_name=run_name,
            wandb_entity=WANDB_ENTITY,
            wandb_config=wandb_config,
            output_dir=f"./validation_outputs_final_vision_llm/{run_name}",
            epochs=EPOCHS,
            max_steps=MAX_STEPS,
            logging_steps=LOGGING_STEPS,
            eval_strategy=EVAL_STRATEGY,
            eval_steps=EVAL_STEPS,
            save_strategy=SAVE_STRATEGY,
            save_steps=SAVE_STEPS,
            batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            optimizer_config=optimizer_config,
            use_separate_lrs=True,
            vision_lr=VISION_LR,
            llm_lr=LLM_LR,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            compute_vqa_accuracy=COMPUTE_VQA_ACCURACY,
            vqa_eval_samples=VQA_EVAL_SAMPLES,
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best val loss: {run_results['best_val_loss']:.4f}")
        if COMPUTE_VQA_ACCURACY:
            print(f"Best VQA accuracy: {run_results['best_vqa_accuracy']:.2%}")
            print(f"Best exact match:  {run_results['best_exact_match']:.2%}")
        print("=" * 70)

    except Exception as e:
        print(f"\nTraining failed: {e}")
        traceback.print_exc()

    finally:
        for var in ("model", "trainer"):
            if var in locals():
                del locals()[var]
        torch.cuda.empty_cache()
        gc.collect()