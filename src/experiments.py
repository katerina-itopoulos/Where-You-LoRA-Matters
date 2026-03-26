import gc

import torch
from tqdm import tqdm

from .data_preprocessing import build_convs_from_rows
from .inference_utils import generate_batch, run_batch_block_vectors
from .vqa_metrics import delta_metrics


def run_pipeline_vectors(
    model: torch.nn.Module,
    processor,
    ds,
    batch_size: int = 1,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Extract pooled visual and text vectors for an entire dataset."""
    all_V0, all_Vproj, all_T = [], [], []

    for s in range(0, len(ds), batch_size):
        e = min(s + batch_size, len(ds))
        V0_l, Vp_l, T_l, _ = run_batch_block_vectors(model, processor, ds, s, e)
        if V0_l:
            all_V0.extend(V0_l)
        all_Vproj.extend(Vp_l)
        all_T.extend(T_l)
        del V0_l, Vp_l, T_l
        gc.collect()
        torch.cuda.empty_cache()

    Vproj_all = torch.cat(all_Vproj, 0)
    T_all = torch.cat(all_T, 0)
    assert Vproj_all.shape[0] == T_all.shape[0]

    return torch.cat(all_V0, 0) if all_V0 else None, Vproj_all, T_all


def run_pipeline_answers(
    model: torch.nn.Module,
    processor,
    ds,
    batch_size: int = 1,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 50,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, list[str]]:
    """Extract pooled vectors and generate answers for an entire dataset."""
    all_V0, all_Vproj, all_T, answers = [], [], [], []

    for s in tqdm(range(0, len(ds), batch_size), desc="Processing batches"):
        e = min(s + batch_size, len(ds))
        V0_l, Vp_l, T_l, ans_full = run_batch_block_vectors(
            model, processor, ds, s, e,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        if V0_l:
            all_V0.extend(V0_l)
        all_Vproj.extend(Vp_l)
        all_T.extend(T_l)
        answers.extend(ans_full)
        del V0_l, Vp_l, T_l, ans_full
        gc.collect()
        torch.cuda.empty_cache()

    Vproj_all = torch.cat(all_Vproj, 0)
    T_all = torch.cat(all_T, 0)
    assert Vproj_all.dim() == 2 and T_all.dim() == 2
    assert Vproj_all.shape[0] == T_all.shape[0]

    return torch.cat(all_V0, 0) if all_V0 else None, Vproj_all, T_all, answers


def run_pipeline_answers_masked(
    model: torch.nn.Module,
    processor,
    ds,
    batch_size: int = 1,
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    repetition_penalty: float = 1.0,
) -> tuple[dict[str, list[str]], list, dict[str, float]]:
    """
    Run full, text-only, and image-only generation for modality collapse probing.

    Returns:
        Tuple of ``(answers_pack, golds, deltas)`` where ``answers_pack`` contains
        predictions under each condition, ``golds`` are the reference answers, and
        ``deltas`` are the delta metrics from ``delta_metrics``.
    """
    ans_full, ans_text, ans_image, golds = [], [], [], []

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    for s in range(0, len(ds), batch_size):
        e = min(s + batch_size, len(ds))
        rows = [ds[i] for i in range(s, e)]

        convs_full_msgs, _, _ = build_convs_from_rows(rows, mode="full")
        convs_text_msgs, _, _ = build_convs_from_rows(rows, mode="text_only")
        convs_image_msgs, _, _ = build_convs_from_rows(rows, mode="image_only")

        convs_full = [[m] for m in convs_full_msgs]
        convs_text = [[m] for m in convs_text_msgs]
        convs_image = [[m] for m in convs_image_msgs]

        ans_full.extend(generate_batch(model, processor, convs_full, **generation_kwargs))
        ans_text.extend(generate_batch(model, processor, convs_text, **generation_kwargs))
        ans_image.extend(generate_batch(model, processor, convs_image, **generation_kwargs))

        for ex in rows:
            golds.append(ex.get("answers") or ex.get("multiple_choice_answer"))

        del (rows, convs_full_msgs, convs_text_msgs, convs_image_msgs,
             convs_full, convs_text, convs_image)
        gc.collect()
        torch.cuda.empty_cache()

    answers_pack = {"full": ans_full, "text_only": ans_text, "image_only": ans_image}
    return answers_pack, golds, delta_metrics(golds, ans_full, ans_text, ans_image)