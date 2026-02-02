# torch version needs to be 2.8.0
import sys
import os
import torch
import numpy as np
import random
from datetime import datetime

# -------------------------------------------------------------------
# Add repo root to sys.path BEFORE importing from `src.*`
# -------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Assumes this file lives in:  <repo_root>/scripts/qwen_gradients_probe.py
sys.path.insert(0, REPO_ROOT)

os.environ['HF_DATASETS_CACHE'] = '/opt/dlami/nvme/hf_cache'

from transformers import AutoProcessor, set_seed
from datasets import load_dataset

from src.finetuning_utils import create_lora_config_vl, VLDataCollatorPadTorch
from src.model_utils import setup_vl_model_and_processor


###############################################################################
# CONFIG
###############################################################################
PLACEMENT_STRATEGY = "projector_vision_grad_probe"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/preprocessed_vqa_lowres_with_instruction"

RANDOM_SEED = 42
MIN_PIXELS = 196 * 32 * 32
MAX_PIXELS = 196 * 32 * 32

PROJECTOR_RANK = 64
LORA_DROPOUT = 0.05
LORA_ALPHA_MULTIPLIER = 2

# Vision attention modules – we'll use fully-qualified names to be extra safe
TARGET_MODULES_VISION_2 = [
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

TARGET_MODULES_PROJECTOR = [
    "visual.merger.linear_fc1", "visual.merger.linear_fc2",
    "visual.deepstack_merger_list.0.linear_fc1", "visual.deepstack_merger_list.0.linear_fc2",
    "visual.deepstack_merger_list.1.linear_fc1", "visual.deepstack_merger_list.1.linear_fc2",
    "visual.deepstack_merger_list.2.linear_fc1", "visual.deepstack_merger_list.2.linear_fc2",
]

# just a few samples needed for gradient sanity
TRAIN_SIZE = 16

LOG_DIR = "./logs"


###############################################################################
# UTILS
###############################################################################
def set_random_seed(seed: int):
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


###############################################################################
# GRADIENT PROBE
###############################################################################
def probe_gradients(model, batch, device="cuda"):
    """
    Run a single forward + backward pass and report gradient stats
    for projector vs vision LoRA parameters.
    """
    model.train()
    batch = {k: v.to(device) for k, v in batch.items()}

    print("\n=== Running single forward + backward for gradient probe ===")
    out = model(**batch)
    loss = out.loss
    print(f"Loss on probe batch: {loss.item():.4f}")

    model.zero_grad()
    loss.backward()

    vision_grads = []
    proj_grads = []
    other_grads = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "lora" not in name:
            continue

        grad_mean = p.grad.abs().mean().item()

        if "visual.blocks" in name:
            vision_grads.append(grad_mean)
        elif "visual.merger" in name or "visual.deepstack" in name:
            proj_grads.append(grad_mean)
        else:
            other_grads.append(grad_mean)

    def safe_stats(arr, label):
        if not arr:
            print(f"{label}: no matching LoRA params with gradients.")
        else:
            arr = np.array(arr)
            print(
                f"{label}: n={len(arr)}, mean={arr.mean():.6e}, "
                f"min={arr.min():.6e}, max={arr.max():.6e}"
            )

    print("\n=== Gradient stats (|grad| mean over LoRA params) ===")
    safe_stats(vision_grads, "Vision LoRA")
    safe_stats(proj_grads, "Projector LoRA")
    safe_stats(other_grads, "Other LoRA (LLM etc.)")
    print("=====================================================\n")


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(
        LOG_DIR,
        f"vision_projector_grad_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_file = open(log_filename, "w")
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"[LOGGING] All prints will also go to: {log_filename}")

    set_random_seed(RANDOM_SEED)

    print("=" * 70)
    print("VISION + PROJECTOR GRADIENT PROBE")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Strategy: {PLACEMENT_STRATEGY}")
    print(f"Train samples (for probe): {TRAIN_SIZE}")
    print("=" * 70)

    ###########################################################################
    # Load processor & tiny subset of train dataset
    ###########################################################################
    print("\n" + "=" * 70)
    print("LOADING PREPROCESSED DATASETS FROM GCS (SMALL SUBSET)")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )

    print("Loading small train subset...")
    train_dataset = load_dataset(
        "parquet",
        data_files=f"gs://{GCS_BUCKET}/{GCS_PREFIX}/train/train_shard_*.parquet",
        split="train",
    ).select(range(TRAIN_SIZE))

    train_cols = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    train_cols = [c for c in train_cols if c in train_dataset.column_names]
    train_dataset.set_format(type="torch", columns=train_cols)

    print(f"✓ Train (probe subset): {len(train_dataset)} samples")

    ###########################################################################
    # Setup model with LoRA on projector + vision
    ###########################################################################
    print("\n" + "=" * 70)
    print("SETTING UP MODEL - PROJECTOR + VISION LORA")
    print("=" * 70)

    target_modules = TARGET_MODULES_PROJECTOR + TARGET_MODULES_VISION_2

    lora_config = create_lora_config_vl(
        lora_r=PROJECTOR_RANK,
        lora_alpha=PROJECTOR_RANK * LORA_ALPHA_MULTIPLIER,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
    )

    model, _, trainable_params, total_params = setup_vl_model_and_processor(
        model_name=MODEL_NAME,
        lora_config=lora_config,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # --- AGGRESSIVE PATCH: force vision tower to run with gradients ---
    print("\n=== APPLYING AGGRESSIVE VISION GRADIENT PATCH ===")
    
    try:
        # Try to access the vision module
        if hasattr(model, 'base_model'):
            visual = model.base_model.model.model.visual
        else:
            visual = model.model.visual
            
        print(f"[PATCH] Found visual module: {type(visual)}")
        
        # Method 1: Override forward to ensure no detach
        orig_forward = visual.forward
        def visual_forward_with_grad(*args, **kwargs):
            # Force enable_grad context
            with torch.enable_grad():
                out = orig_forward(*args, **kwargs)
            # If output is still detached somehow, clone it
            if isinstance(out, torch.Tensor) and not out.requires_grad:
                print("[PATCH WARNING] Vision output was detached! Cloning with grad")
                out = out.clone().requires_grad_(True)
            return out
        visual.forward = visual_forward_with_grad
        print("[PATCH] Wrapped visual.forward with torch.enable_grad()")
        
        # Method 2: Explicitly set requires_grad on all vision params
        vision_param_count = 0
        for name, param in visual.named_parameters():
            if 'lora' in name:
                param.requires_grad_(True)
                vision_param_count += 1
        print(f"[PATCH] Set requires_grad=True for {vision_param_count} vision LoRA params")
        
        # Method 3: Disable gradient checkpointing on vision if present
        if hasattr(visual, 'gradient_checkpointing'):
            visual.gradient_checkpointing = False
            print("[PATCH] Disabled vision gradient checkpointing")
        
        # Method 4: Set visual module to train mode explicitly
        visual.train()
        print("[PATCH] Set visual.train()")
            
        print("[PATCH] ✅ Vision gradient patch applied successfully!")
        
    except Exception as e:
        print(f"\n[PATCH FAILED] ❌ Could not patch visual module: {e}")
        print("Vision gradients may not flow!")
        import traceback
        traceback.print_exc()
    # --- END PATCH ---

    model.config.use_cache = False
    # DON'T enable gradient checkpointing - it breaks vision gradients!
    # model.gradient_checkpointing_enable()  # <-- DISABLED
    print("[INFO] Gradient checkpointing DISABLED to allow vision gradients")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(
        f"✓ Trainable params: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Debug: trainable param names and counts by region
    print("\n=== DEBUG: Trainable Parameter Names (LoRA only) ===")
    vision_count = 0
    proj_count = 0
    llm_count = 0
    other_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name:
            print(name)
            if "visual.blocks" in name:
                vision_count += 1
            elif "visual.merger" in name or "visual.deepstack" in name:
                proj_count += 1
            elif "language_model" in name:
                llm_count += 1
            else:
                other_count += 1

    print(f"\nMatched trainable LoRA params:")
    print(f"  Vision: {vision_count} (should be >0 for this probe)")
    print(f"  Projector: {proj_count} (should be >0)")
    print(f"  LLM: {llm_count} (depending on config, can be 0)")
    print(f"  Other (unmatched): {other_count}")
    print("=" * 50)

    ###########################################################################
    # Take a single batch and probe gradients (with real collator)
    ###########################################################################
    print("\nTaking one batch from train subset for gradient probe...")
    data_collator = VLDataCollatorPadTorch()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=data_collator,
    )
    batch = next(iter(train_loader))

    probe_gradients(model, batch, device=device)

    print("\nGradient probe complete. No training was run.")
    print(f"Log file: {log_filename}")