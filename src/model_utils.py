import torch
from peft import PeftModel
from typing import Optional, Tuple
from peft import get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

def setup_vl_model_and_processor(model_name, lora_config, min_pixels=256*32*32, max_pixels=1280*32*32):
    """
    Load vision-language model, apply LoRA
    (Processor should be loaded separately before calling this)

    Args:
        model_name: HuggingFace model name
        lora_config: LoraConfig object
        min_pixels: Min image resolution
        max_pixels: Max image resolution

    Returns:
        model, processor, trainable_params, total_params
    """
    print(f"\n📦 Loading vision-language model...")
    print(f"   Model: {model_name}")

    # Load processor (handles both images and text)
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    # IMPORTANT: Set model to train mode before applying LoRA
    model.train()

    # Apply LoRA
    print(f"\n⚙️  Applying LoRA (rank={lora_config.r}, alpha={lora_config.lora_alpha})...")
    model = get_peft_model(model, lora_config)

    # Ensure model is in training mode after LoRA
    model.train()

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"✓ Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")

    # Verify gradients are enabled
    has_grad = any(p.requires_grad for p in model.parameters())
    if not has_grad:
        raise RuntimeError("No parameters have requires_grad=True! LoRA may not be applied correctly.")

    return model, processor, trainable_params, total_params

def load_vl_model_with_lora(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    merge_lora: bool = False,
    attn_implementation: str = "sdpa",
    min_pixels: int = 196*32*32,  # 448×448
    max_pixels: int = 196*32*32
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """
    Load Qwen3-VL model with optional LoRA adapter for inference.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-VL-8B-Instruct")
        checkpoint_path: Path to LoRA checkpoint
                        - None: load base model only
                        - "wandb:artifact_name": download from W&B
                        - "gs://...": GCS path
                        - "outputs/...": local path
        device: Device to load on
        merge_lora: If True, merge LoRA into base model
        attn_implementation: "sdpa" or "flash_attention_2"
        min_pixels: Min image resolution
        max_pixels: Max image resolution
        eval_mode: Set model to eval mode (disable dropout)
    
    Returns:
        (model, processor)
    """
    print(f"\n{'='*70}")
    print("🔧 Loading Qwen3-VL Model for Inference")
    print(f"{'='*70}")
    print(f"Base model: {model_name}")
    print(f"LoRA checkpoint: {checkpoint_path or 'None (base model only)'}")
    print(f"Attention: {attn_implementation}")
    print(f"Resolution: {int((min_pixels/(32*32))**0.5)}×{int((max_pixels/(32*32))**0.5)} pixels")
    print(f"{'='*70}\n")
    
    # Handle W&B artifacts
    if checkpoint_path and checkpoint_path.startswith("wandb:"):
        artifact_name = checkpoint_path.replace("wandb:", "")
        print("📦 Downloading from W&B Artifacts...")
        from src.wandb_utils import download_lora_from_wandb
        checkpoint_path = download_lora_from_wandb(artifact_name)
    
    # Load processor with resolution settings
    print("📝 Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True
    )
    
    # Load base model
    print(" Loading base Qwen3-VL model...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn_implementation,
        trust_remote_code=True
    )
    
    # Load LoRA if provided
    if checkpoint_path:
        print(f"🔗 Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        if merge_lora:
            print("🔀 Merging LoRA into base model...")
            model = model.merge_and_unload()
            print("✓ LoRA merged! Model is standalone.")
        else:
            print("✓ LoRA loaded as adapter.")
    else:
        print("✓ Using base model (no LoRA)")
        model = base_model
    
    # Set eval mode for inference
    model.eval()
    print("✓ Model in eval mode")
    
    print(f"\n{'='*70}")
    print("✅ Qwen3-VL Model Ready for Inference!")
    print(f"{'='*70}\n")
    
    return model, processor