from typing import Dict, List, Optional
import torch
from PIL import Image

from .data_preprocessing import build_convs_from_rows


def align_for_model(model, processor, inputs: dict) -> dict:

    try:
        emb_dev = model.get_input_embeddings().weight.device
    except Exception:
        emb_dev = next(model.parameters()).device

    for k in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        v = inputs.get(k, None)
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(emb_dev, non_blocking=True)

    visual_mod = getattr(model, "visual", model)
    first_param = next(visual_mod.parameters())
    vis_dev   = first_param.device
    vis_dtype = getattr(visual_mod, "dtype", first_param.dtype)

    pv = inputs.get("pixel_values", None)
    if isinstance(pv, torch.Tensor):
        inputs["pixel_values"] = pv.to(vis_dev, dtype=vis_dtype, non_blocking=True)

    grid = inputs.get("image_grid_thw", None)
    if isinstance(grid, torch.Tensor):
        inputs["image_grid_thw"] = grid.to(vis_dev, non_blocking=True)

    return inputs

def _coerce_visual_tokens(Vraw, B: int, D_llm: int):
    """
    Normalize various visual outputs to [B, T, D_llm].
    Handles:
      - tensor [B, T, D_llm]
      - tensor [T, D_llm] when B==1
      - tuple/list where the first tensor is [T, D_llm] or [B, T, D_llm]
    """
    import torch

    def _as_BTD(x):
        if not torch.is_tensor(x):
            return None
        if x.dim() == 3:
            if x.shape[0] == B and x.shape[-1] == D_llm:
                return x
            if x.shape[1] == B and x.shape[-1] == D_llm:
                return x.permute(1,0,2).contiguous()
        if x.dim() == 2 and x.shape[-1] == D_llm:
            if B == 1:
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
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x[:, start_idx:, :].mean(dim=1)

def forward_internals(model, processor, convs):
    inputs = processor.apply_chat_template(
        convs, add_generation_prompt=False, tokenize=True,
        return_tensors="pt", return_dict=True
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
            if "image_grid_thw" in inputs:
                Vraw = model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
            else:
                Vraw = model.visual(inputs["pixel_values"])
        Vproj = _coerce_visual_tokens(Vraw, B, D_llm)  # -> [B, T_img, D_llm] or None

    return H, Vproj, inputs

def pool_vectors(V0: Optional[torch.Tensor], Vproj: torch.Tensor, H: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    V0:    [B, Ti0, D] or None
    Vproj: [B, Ti,  D]
    H:     [B, T_all, D] (fused LLM states with image tokens first)
    Returns CPU tensors: V0_vec, Vproj_vec, T_fused_vec (each [B, D])
    """
    out: Dict[str, torch.Tensor] = {}
    if V0 is not None:
        out["V0_vec"] = V0.mean(dim=1).to("cpu", non_blocking=True)

    out["Vproj_vec"] = Vproj.mean(dim=1).to("cpu", non_blocking=True)

    Ti = int(Vproj.shape[1]) if Vproj is not None else 0
    B, S = H.shape[:2]
    T_span = H[:, Ti:, :] if Ti < S else H  # fallback if no text tokens
    out["T_fused_vec"] = T_span.mean(dim=1).to("cpu", non_blocking=True)

    del H, Vproj, V0
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return out

def generate_answers(
    model,
    processor,
    imgs: List[Image.Image],
    qs: List[str],
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
) -> List[str]:
    import torch, gc
    outs: List[str] = []

    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    for img, q in zip(imgs, qs):
        conv = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": q},
        ]}]
        gen_inp = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True
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
        gc.collect(); torch.cuda.empty_cache()
    return outs

def generate_batch(
    model,
    processor,
    convs: List[List[Dict]],
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    repetition_penalty: float = 1.0,
) -> List[str]:
    import torch, gc
    outs: List[str] = []

    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    for conv in convs:
        gen_inp = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True
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
        gc.collect(); torch.cuda.empty_cache()
    return outs


def run_batch_block_vectors(
    model, processor, ds, start: int, end: int,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 50,
):
    import torch, gc

    batch = ds.select(list(range(start, end)))
    rows  = [batch[i] for i in range(len(batch))]
    if not rows:
        return [], [], [], []

    convs_full, imgs, qs = build_convs_from_rows(rows, mode="full")

    #tokenize multimodal chats
    inputs = processor.apply_chat_template(
        convs_full, add_generation_prompt=False, tokenize=True,
        return_tensors="pt", return_dict=True
    )
    inputs = align_for_model(model, processor, inputs)

    #fused LLM states (last layer)
    with torch.inference_mode():
        outs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        H = outs.hidden_states[-1]

    B, _, D_llm = H.shape
    with torch.inference_mode():
        Vraw = (model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
                if "image_grid_thw" in inputs else
                model.visual(inputs["pixel_values"]))

    Vproj = _coerce_visual_tokens(Vraw, B, D_llm)

    if Vproj is None:
        print(f"[warn] skip block {start}:{end} (cannot coerce visual to [B,T,{D_llm}])")
        return [], [], [], []

    #pool to per-sample vectors on CPU
    Ti = int(Vproj.shape[1])                                  # image token count inserted first
    Vproj_vec   = Vproj.mean(1).to("cpu", non_blocking=True)  # [B, D_llm]
    T_span      = H[:, Ti:, :] if Ti < H.shape[1] else H
    T_fused_vec = T_span.mean(1).to("cpu", non_blocking=True) # [B, D_llm]
    V0_vec_list = []  # no pre-proj on Qwen path

    assert Vproj_vec.shape[0] == T_fused_vec.shape[0], "batch mismatch after pooling"

    #free GPU before generation
    del outs, H, Vproj, inputs
    gc.collect(); torch.cuda.empty_cache()

    #GENERATION (FULL only)
    answers_full = generate_answers(
        model, processor, imgs, qs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    # return lists (each entry is a [B, D] tensor on CPU)
    return V0_vec_list, [Vproj_vec], [T_fused_vec], answers_full