import torch
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

    # Load vision-language model with Flash Attention 2
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