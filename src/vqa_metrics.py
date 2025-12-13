"""
Modality collapse metrics with VQA-specific accuracy computation
"""

def normalize_answer(answer):
    """Normalize VQA answers (lowercase, strip, remove articles)"""
    if answer is None:
        return ""
    
    answer = str(answer).lower().strip()
    
    # Remove punctuation
    answer = answer.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    
    # Remove articles
    for article in ['a', 'an', 'the']:
        answer = answer.replace(f' {article} ', ' ')
        if answer.startswith(f'{article} '):
            answer = answer[len(article)+1:]
    
    return answer.strip()


def vqa_match(pred, gold):
    """
    VQA accuracy: min(matching_annotations / 3, 1.0)
    
    Args:
        pred: predicted answer (string)
        gold: reference - either list of human answers or dict with 'answer' key
    
    Returns:
        accuracy score between 0 and 1
    """
    # Extract human answers
    if gold is None:
        return 0.0
    
    if isinstance(gold, list):
        # Already a list of answers
        if gold and isinstance(gold[0], dict) and "answer" in gold[0]:
            human_answers = [g["answer"] for g in gold]
        else:
            human_answers = gold
    elif isinstance(gold, dict) and "answer" in gold:
        human_answers = gold["answer"] if isinstance(gold["answer"], list) else [gold["answer"]]
    else:
        human_answers = [gold]
    
    # Normalize prediction
    pred_normalized = normalize_answer(pred)
    
    # Count matching human answers
    matching_count = sum(
        1 for ans in human_answers 
        if normalize_answer(ans) == pred_normalized
    )
    
    # VQA accuracy formula: min(count / 3, 1)
    return min(matching_count / 3.0, 1.0)


def compute_vqa_accuracy(predictions, references):
    """
    Compute VQA accuracy over a batch
    
    Args:
        predictions: list of predicted answers (strings)
        references: list of gold answers (format varies, see vqa_match)
    
    Returns:
        average VQA accuracy
    """
    if not predictions or not references:
        return 0.0
    
    total_score = sum(
        vqa_match(pred, ref) 
        for pred, ref in zip(predictions, references)
    )
    
    return total_score / len(predictions)


def delta_metrics(golds, full, text_only, image_only):
    """
    Compute delta metrics for modality collapse analysis using VQA accuracy
    
    Args:
        golds: list of gold answers
        full: predictions with both text and image
        text_only: predictions with text only (image ablated)
        image_only: predictions with image only (text ablated)
    
    Returns:
        dict with accuracy and delta metrics
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
    
    # Compute VQA accuracy for each condition
    acc_full = sum(vqa_match(p, g) for p, g in zip(full, golds)) / N
    acc_txt = sum(vqa_match(p, g) for p, g in zip(text_only, golds)) / N
    acc_img = sum(vqa_match(p, g) for p, g in zip(image_only, golds)) / N
    
    return {
        "Acc_full": acc_full,
        "Acc_textOnly": acc_txt,
        "Acc_imageOnly": acc_img,
        "Delta_img": acc_full - acc_txt,  # How much image helps
        "Delta_txt": acc_full - acc_img,  # How much text helps
    }