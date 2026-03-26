def normalize_answer(answer: str | None) -> str:
    """Lowercase, strip punctuation and articles from a VQA answer string."""
    if answer is None:
        return ""

    answer = str(answer).lower().strip()
    answer = answer.replace(".", "").replace(",", "").replace("!", "").replace("?", "")

    for article in ["a", "an", "the"]:
        answer = answer.replace(f" {article} ", " ")
        if answer.startswith(f"{article} "):
            answer = answer[len(article) + 1:]

    return answer.strip()


def vqa_match(pred: str, gold: list | dict | str | None) -> float:
    """
    Compute VQA accuracy for a single prediction against reference annotations.

    Args:
        pred: Predicted answer string.
        gold: Reference answers as a list of strings, list of dicts with an
              ``"answer"`` key, a single dict, or a plain string.

    Returns:
        Score in ``[0, 1]`` via ``min(matching_annotations / 3, 1.0)``.
    """
    if gold is None:
        return 0.0

    if isinstance(gold, list):
        if gold and isinstance(gold[0], dict) and "answer" in gold[0]:
            human_answers = [g["answer"] for g in gold]
        else:
            human_answers = gold
    elif isinstance(gold, dict) and "answer" in gold:
        human_answers = gold["answer"] if isinstance(gold["answer"], list) else [gold["answer"]]
    else:
        human_answers = [gold]

    pred_normalized = normalize_answer(pred)
    matching_count = sum(1 for ans in human_answers if normalize_answer(ans) == pred_normalized)

    return min(matching_count / 3.0, 1.0)


def compute_vqa_accuracy(predictions: list[str], references: list) -> float:
    """Return mean VQA accuracy over a batch of predictions and references."""
    if not predictions or not references:
        return 0.0

    return sum(vqa_match(p, r) for p, r in zip(predictions, references)) / len(predictions)


def delta_metrics(
    golds: list,
    full: list[str],
    text_only: list[str],
    image_only: list[str],
) -> dict[str, float]:
    """
    Compute per-condition VQA accuracy and modality contribution deltas.

    Args:
        golds: Gold reference answers.
        full: Predictions from the full multimodal model.
        text_only: Predictions with the image ablated.
        image_only: Predictions with the text ablated.

    Returns:
        Dict with ``Acc_full``, ``Acc_textOnly``, ``Acc_imageOnly``,
        ``Delta_img`` (image contribution), and ``Delta_txt`` (text contribution).
    """
    N = len(golds)
    if N == 0:
        return {
            "Acc_full": 0.0,
            "Acc_textOnly": 0.0,
            "Acc_imageOnly": 0.0,
            "Delta_img": 0.0,
            "Delta_txt": 0.0,
        }

    acc_full = sum(vqa_match(p, g) for p, g in zip(full, golds)) / N
    acc_txt = sum(vqa_match(p, g) for p, g in zip(text_only, golds)) / N
    acc_img = sum(vqa_match(p, g) for p, g in zip(image_only, golds)) / N

    return {
        "Acc_full": acc_full,
        "Acc_textOnly": acc_txt,
        "Acc_imageOnly": acc_img,
        "Delta_img": acc_full - acc_txt,
        "Delta_txt": acc_full - acc_img,
    }