import torch
from peft import PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def setup_vl_model_and_processor(
    model_name: str,
    lora_config,
    min_pixels: int = 256 * 32 * 32,
    max_pixels: int = 1280 * 32 * 32,
) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor, int, int]:
    """
    Load a Qwen3-VL model and processor, apply LoRA, and return training parameter counts.

    Args:
        model_name: HuggingFace model name.
        lora_config: Configured ``LoraConfig`` object.
        min_pixels: Minimum image resolution in pixels.
        max_pixels: Maximum image resolution in pixels.

    Returns:
        Tuple of ``(model, processor, trainable_params, total_params)``.
    """
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model.train()
    model = get_peft_model(model, lora_config)
    model.train()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No parameters have requires_grad=True. LoRA may not be applied correctly.")

    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, processor, trainable_params, total_params


def load_vl_model_with_lora(
    model_name: str,
    checkpoint_path: str | None = None,
    device: str = "cuda",
    merge_lora: bool = False,
    attn_implementation: str = "sdpa",
    min_pixels: int = 196 * 32 * 32,
    max_pixels: int = 196 * 32 * 32,
) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """
    Load a Qwen3-VL model with an optional LoRA adapter for inference.

    Args:
        model_name: HuggingFace model name.
        checkpoint_path: LoRA checkpoint source. Accepts ``None`` (base model
            only), a ``"wandb:<artifact>"`` string, a GCS path, or a local path.
        device: Device string passed to ``device_map``.
        merge_lora: When ``True``, merges the adapter into base model weights.
        attn_implementation: Attention backend, ``"sdpa"`` or ``"flash_attention_2"``.
        min_pixels: Minimum image resolution in pixels.
        max_pixels: Maximum image resolution in pixels.

    Returns:
        Tuple of ``(model, processor)``.
    """
    if checkpoint_path and checkpoint_path.startswith("wandb:"):
        from src.wandb_utils import download_lora_from_wandb
        checkpoint_path = download_lora_from_wandb(checkpoint_path.replace("wandb:", ""))

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )

    if checkpoint_path:
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()

    return model, processor