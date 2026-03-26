import gc

import torch
from PIL import Image

from .data_preprocessing import build_convs_from_rows


def align_for_model(model: torch.nn.Module, processor, inputs: dict) -> dict:
    """Move input tensors to the correct devices and dtypes for the given model."""
    try:
        emb_dev = model.get_input_embeddings().weight.device
    except Exception:
        emb_dev = next(model.parameters()).device

    for k in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        v = inputs.get(k)
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(emb_dev, non_blocking=True)

    visual_mod = getattr(model, "visual", model)
    first_param = next(visual_mod.parameters())
    vis_dev = first_param.device
    vis_dtype = getattr(visual_mod, "dtype", first_param.dtype)

    pv = inputs.get("pixel_values")
    if isinstance(pv, torch.Tensor):
        inputs["pixel_values"] = pv.to(vis_dev, dtype=vis_dtype, non_blocking=True)

    grid = inputs.get("image_grid_thw")
    if isinstance(grid, torch.Tensor):
        inputs["image_grid_thw"] = grid.to(vis_dev, non_blocking=True)

    return inputs


def _coerce_visual_tokens(
    Vraw: torch.Tensor | tuple | list,
    B: int,
    D_llm: int,
) -> torch.Tensor | None:
    """Normalize visual encoder output to shape ``[B, T, D_llm]``, or return ``None``."""
    def _as_BTD(x: torch.Tensor) -> torch.Tensor | None:
        if not torch.is_tensor(x):
            return None
        if x.dim() == 3:
            if x.shape[0] == B and x.shape[-1] == D_llm:
                return x
            if x.shape[1] == B and x.shape[-1] == D_llm:
                return x.permute(1, 0, 2).contiguous()
        if x.dim() == 2 and x.shape[-1] == D_llm and B == 1:
            return x.unsqueeze(0)
        return None

    cand = _as_BTD(Vraw)
    if cand is not None:
        return cand

    if isinstance(Vraw, (list, tuple)):
        for t in Vraw:
            cand = _as_BTD(t)
            if cand is not None:
                return cand

    return None


def mean_pool_slice(x: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """Mean-pool a tensor over the sequence dimension from ``start_idx`` onward."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x[:, start_idx:, :].mean(dim=1)


def forward_internals(
    model: torch.nn.Module,
    processor,
    convs: list[list[dict]],
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """
    Run a forward pass and return the last LLM hidden state, projected visual tokens, and inputs.

    Returns:
        Tuple of ``(H, Vproj, inputs)`` where ``H`` is ``[B, T, D]``,
        ``Vproj`` is ``[B, T_img, D]`` or ``None``, and ``inputs`` is the
        processed input dict.
    """
    inputs = processor.apply_chat_template(
        convs,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = align_for_model(model, processor, inputs)

    with torch.inference_mode():
        outs = model(
            **inputs,
            output_hidden_states=True,
            output_image_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    H = outs.hidden_states[-1]
    B, _, D_llm = H.shape

    Vproj = None
    ihs = getattr(outs, "image_hidden_states", None)
    if ihs:
        for t in reversed(ihs):
            if isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[0] == B and t.shape[-1] == D_llm:
                Vproj = t
                break

    if Vproj is None:
        with torch.inference_mode():
            Vraw = (
                model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
                if "image_grid_thw" in inputs
                else model.visual(inputs["pixel_values"])
            )
        Vproj = _coerce_visual_tokens(Vraw, B, D_llm)

    return H, Vproj, inputs


def pool_vectors(
    V0: torch.Tensor | None,
    Vproj: torch.Tensor,
    H: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Mean-pool visual and text hidden states to per-sample vectors on CPU.

    Args:
        V0: Pre-projection visual states ``[B, Ti0, D]``, or ``None``.
        Vproj: Projected visual states ``[B, Ti, D]``.
        H: Fused LLM hidden states ``[B, T_all, D]`` with image tokens first.

    Returns:
        Dict of CPU tensors with keys ``V0_vec``, ``Vproj_vec``, ``T_fused_vec``,
        each of shape ``[B, D]``.
    """
    out: dict[str, torch.Tensor] = {}

    if V0 is not None:
        out["V0_vec"] = V0.mean(dim=1).to("cpu", non_blocking=True)

    out["Vproj_vec"] = Vproj.mean(dim=1).to("cpu", non_blocking=True)

    Ti = int(Vproj.shape[1])
    B, S = H.shape[:2]
    T_span = H[:, Ti:, :] if Ti < S else H
    out["T_fused_vec"] = T_span.mean(dim=1).to("cpu", non_blocking=True)

    del H, Vproj, V0
    gc.collect()
    torch.cuda.empty_cache()

    return out


def generate_answers(
    model: torch.nn.Module,
    processor,
    imgs: list[Image.Image],
    qs: list[str],
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """Generate one answer per (image, question) pair, clearing GPU memory after each sample."""
    pad_id = getattr(processor.tokenizer, "pad_token_id", None) or processor.tokenizer.eos_token_id
    outs: list[str] = []

    for img, q in zip(imgs, qs):
        conv = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": q},
        ]}]
        gen_inp = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        )
        gen_inp = align_for_model(model, processor, gen_inp)

        with torch.inference_mode():
            gen_ids = model.generate(
                **gen_inp,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=pad_id,
            )

        start = gen_inp["input_ids"].shape[1]
        outs.append(processor.tokenizer.decode(gen_ids[0][start:], skip_special_tokens=True).strip())

        del gen_ids, gen_inp
        gc.collect()
        torch.cuda.empty_cache()

    return outs


def generate_batch(
    model: torch.nn.Module,
    processor,
    convs: list[list[dict]],
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """Generate one answer per conversation, clearing GPU memory after each sample."""
    pad_id = getattr(processor.tokenizer, "pad_token_id", None) or processor.tokenizer.eos_token_id
    outs: list[str] = []

    for conv in convs:
        gen_inp = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        )
        gen_inp = align_for_model(model, processor, gen_inp)

        with torch.inference_mode():
            ids = model.generate(
                **gen_inp,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=pad_id,
            )

        start = gen_inp["input_ids"].shape[1]
        outs.append(processor.tokenizer.decode(ids[0][start:], skip_special_tokens=True).strip())

        del ids, gen_inp
        gc.collect()
        torch.cuda.empty_cache()

    return outs


def run_batch_block_vectors(
    model: torch.nn.Module,
    processor,
    ds,
    start: int,
    end: int,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 50,
) -> tuple[list, list[torch.Tensor], list[torch.Tensor], list[str]]:
    """
    Extract pooled visual and text vectors and generate answers for a dataset slice.

    Returns:
        Tuple of ``(V0_vec_list, [Vproj_vec], [T_fused_vec], answers_full)``.
        ``V0_vec_list`` is always empty on the Qwen path (no pre-projection states).
    """
    batch = ds.select(list(range(start, end)))
    rows = [batch[i] for i in range(len(batch))]
    if not rows:
        return [], [], [], []

    convs_full, imgs, qs = build_convs_from_rows(rows, mode="full")

    inputs = processor.apply_chat_template(
        convs_full,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = align_for_model(model, processor, inputs)

    with torch.inference_mode():
        outs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        H = outs.hidden_states[-1]

    B, _, D_llm = H.shape

    with torch.inference_mode():
        Vraw = (
            model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
            if "image_grid_thw" in inputs
            else model.visual(inputs["pixel_values"])
        )

    Vproj = _coerce_visual_tokens(Vraw, B, D_llm)
    if Vproj is None:
        print(f"Skipping block {start}:{end} — cannot coerce visual output to [B, T, {D_llm}]")
        return [], [], [], []

    Ti = int(Vproj.shape[1])
    Vproj_vec = Vproj.mean(1).to("cpu", non_blocking=True)
    T_span = H[:, Ti:, :] if Ti < H.shape[1] else H
    T_fused_vec = T_span.mean(1).to("cpu", non_blocking=True)

    assert Vproj_vec.shape[0] == T_fused_vec.shape[0], "batch mismatch after pooling"

    del outs, H, Vproj, inputs
    gc.collect()
    torch.cuda.empty_cache()

    answers_full = generate_answers(
        model, processor, imgs, qs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    return [], [Vproj_vec], [T_fused_vec], answers_full