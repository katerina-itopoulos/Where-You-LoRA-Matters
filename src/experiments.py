from .data_preprocessing import build_convs_from_rows
from .inference_utils import generate_batch, run_batch_block_vectors
from .vqa_metrics import delta_metrics

def run_pipeline_vectors(model, processor, ds, batch_size: int = 1):
    import gc, torch
    all_V0, all_Vproj, all_T = [], [], []
    for s in range(0, len(ds), batch_size):
        e = min(s + batch_size, len(ds))
        V0_l, Vp_l, T_l, _ = run_batch_block_vectors(model, processor, ds, s, e)
        if V0_l: all_V0.extend(V0_l)
        all_Vproj.extend(Vp_l); all_T.extend(T_l)
        del V0_l, Vp_l, T_l; gc.collect(); torch.cuda.empty_cache()
    V0_all    = torch.cat(all_V0,    0) if all_V0 else None
    Vproj_all = torch.cat(all_Vproj, 0)
    T_all     = torch.cat(all_T,     0)
    assert Vproj_all.shape[0] == T_all.shape[0]
    return V0_all, Vproj_all, T_all

def run_pipeline_answers(
    model, processor, ds100, batch_size: int = 1,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 50,
):
    import gc, torch
    from tqdm import tqdm  # Add this import
    all_V0, all_Vproj, all_T, answers = [], [], [], []

    # Add tqdm here:
    for s in tqdm(range(0, len(ds100), batch_size), desc="Processing batches"):
        e = min(s + batch_size, len(ds100))
        V0_l, Vp_l, T_l, ans_full = run_batch_block_vectors(
            model, processor, ds100, s, e,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        if V0_l: all_V0.extend(V0_l)
        all_Vproj.extend(Vp_l); all_T.extend(T_l); answers.extend(ans_full)
        del V0_l, Vp_l, T_l, ans_full; gc.collect(); torch.cuda.empty_cache()

    V0_all    = torch.cat(all_V0,    0) if all_V0 else None
    Vproj_all = torch.cat(all_Vproj, 0)
    T_all     = torch.cat(all_T,     0)
    assert Vproj_all.dim()==2 and T_all.dim()==2
    assert Vproj_all.shape[0] == T_all.shape[0]
    return V0_all, Vproj_all, T_all, answers

def run_pipeline_answers_masked(
    model,
    processor,
    ds100,
    batch_size: int = 1,
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    repetition_penalty: float = 1.0,
):
    """
    Functional probe for modality collapse:
      - FULL, TEXT-ONLY, IMAGE-ONLY generations per batch
      - Returns predictions, golds, and Δ metrics (Δ_img, Δ_txt)
    """
    import gc, torch
    ans_full, ans_text, ans_image, golds = [], [], [], []

    for s in range(0, len(ds100), batch_size):
        e = min(s + batch_size, len(ds100))
        rows = [ds100[i] for i in range(s, e)]

        # Build message lists (each item is ONE message dict)
        convs_full_msgs,  _, _ = build_convs_from_rows(rows, mode="full")
        convs_text_msgs,  _, _ = build_convs_from_rows(rows, mode="text_only")
        convs_image_msgs, _, _ = build_convs_from_rows(rows, mode="image_only")

        # Wrap each single-message dict into a conversation (list of messages)
        convs_full  = [[m] for m in convs_full_msgs]
        convs_text  = [[m] for m in convs_text_msgs]
        convs_image = [[m] for m in convs_image_msgs]

        # Batched generation per mode
        full_preds  = generate_batch(
            model, processor, convs_full,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature,
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        text_preds  = generate_batch(
            model, processor, convs_text,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature,
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        image_preds = generate_batch(
            model, processor, convs_image,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature,
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        ans_full.extend(full_preds)
        ans_text.extend(text_preds)
        ans_image.extend(image_preds)

        # Golds: VQAv2-style list of dicts or string (MC datasets)
        for ex in rows:
            golds.append(ex.get("answers") or ex.get("multiple_choice_answer"))

        del (rows, convs_full_msgs, convs_text_msgs, convs_image_msgs,
             convs_full, convs_text, convs_image,
             full_preds, text_preds, image_preds)
        gc.collect(); torch.cuda.empty_cache()

    deltas = delta_metrics(golds, ans_full, ans_text, ans_image)
    answers_pack = {"full": ans_full, "text_only": ans_text, "image_only": ans_image}
    return answers_pack, golds, deltas