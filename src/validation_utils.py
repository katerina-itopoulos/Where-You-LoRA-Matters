# src/validation_utils.py

import numpy as np
import torch
import wandb
from transformers import TrainerCallback
import itertools


def generate_validation_configs(lora_ranks, learning_rates, lora_alpha_multiplier):
    """
    Generate all combinations of hyperparameters for validation.
    
    Args:
        lora_ranks: List of LoRA ranks to try
        learning_rates: List of learning rates to try
        lora_alpha_multiplier: Multiplier for LoRA alpha (alpha = rank * multiplier)
    
    Yields:
        Dict with hyperparameter configuration
    """
    for rank, lr in itertools.product(lora_ranks, learning_rates):
        yield {
            "lora_r": rank,
            "lora_alpha": rank * lora_alpha_multiplier,
            "learning_rate": lr,
        }


def create_validation_run_name(rank, lr, run_id):
    """Create a descriptive run name for W&B."""
    return f"r{rank}_lr{lr:.0e}_run{run_id}"


class VQAAccuracyCallback(TrainerCallback):
    """
    Custom callback to compute both VQA accuracy and exact match accuracy.
    
    Metrics computed:
    - VQA Accuracy: Official VQA metric (min(count/3, 1.0))
    - Exact Match: Simple exact match with most common answer
    """
    
    def __init__(self, processor, eval_dataset, max_new_tokens=16, eval_samples=500):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.max_new_tokens = max_new_tokens
        self.eval_samples = eval_samples
        
        # Track best for both metrics
        self.best_vqa_accuracy = 0.0
        self.best_exact_match = 0.0
    
    def compute_vqa_score(self, prediction, answer_counts):
        """
        Official VQA accuracy metric.
        Score = min(#annotators_that_said_answer / 3, 1.0)
        
        Args:
            prediction: Model's predicted answer (string)
            answer_counts: Dict of {answer: count} from annotators
        
        Returns:
            Accuracy score [0, 1]
        """
        pred_norm = prediction.lower().strip()
        
        for answer, count in answer_counts.items():
            ans_norm = answer.lower().strip()
            if pred_norm == ans_norm:
                return min(count / 3.0, 1.0)
        
        return 0.0
    
    def compute_exact_match(self, prediction, most_common_answer):
        """
        Simple exact match accuracy.
        
        Args:
            prediction: Model's predicted answer
            most_common_answer: Most common answer from annotators
        
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred_norm = prediction.lower().strip()
        ans_norm = most_common_answer.lower().strip()
        return 1.0 if pred_norm == ans_norm else 0.0
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """
        Run after each evaluation to compute both VQA and exact match accuracy.
        """
        import json
        from tqdm import tqdm
        
        print(f"\n{'='*70}")
        print("🎯 Computing Accuracy Metrics on Validation Set")
        print(f"{'='*70}")
        
        model.eval()
        
        # Separate scores for each metric
        total_vqa_score = 0.0
        total_exact_match = 0.0
        num_samples = 0
        
        # Determine evaluation size
        if self.eval_samples is None:
            eval_size = len(self.eval_dataset)
            indices = range(eval_size)
            print(f"Evaluating on ALL {eval_size} validation samples")
        else:
            eval_size = min(self.eval_samples, len(self.eval_dataset))
            indices = np.random.choice(len(self.eval_dataset), eval_size, replace=False)
            print(f"Evaluating on {eval_size} random samples")
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Generating answers"):
                sample = self.eval_dataset[int(idx)]
                
                # Get ground truth
                if "answer_counts" in sample:
                    answer_counts_raw = sample["answer_counts"]
                    # Parse JSON string to dict
                    if isinstance(answer_counts_raw, str):
                        answer_counts = json.loads(answer_counts_raw)
                    else:
                        answer_counts = answer_counts_raw
                    
                    # Get most common answer
                    most_common_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                else:
                    # Fallback
                    most_common_answer = sample.get('answer', sample.get('labels', ''))
                    answer_counts = {most_common_answer: 10}
                
                # Prepare input for generation
                inputs = {
                    'input_ids': sample['input_ids'].unsqueeze(0).to(model.device),
                    'attention_mask': sample['attention_mask'].unsqueeze(0).to(model.device),
                    'pixel_values': sample['pixel_values'].unsqueeze(0).to(model.device),
                    'image_grid_thw': sample['image_grid_thw'].unsqueeze(0).to(model.device),
                }
                
                # Generate answer with DETERMINISTIC settings
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    
                    # Deterministic generation (for consistent evaluation)
                    do_sample=False,
                    
                    # Standard settings
                    repetition_penalty=1.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                
                # Decode (only new tokens)
                start_idx = inputs['input_ids'].shape[1]
                prediction = self.processor.tokenizer.decode(
                    generated_ids[0][start_idx:], 
                    skip_special_tokens=True
                ).strip()
                
                # Compute BOTH metrics
                vqa_score = self.compute_vqa_score(prediction, answer_counts)
                exact_match_score = self.compute_exact_match(prediction, most_common_answer)
                
                total_vqa_score += vqa_score
                total_exact_match += exact_match_score
                num_samples += 1
        
        # Compute averages
        vqa_accuracy = total_vqa_score / num_samples if num_samples > 0 else 0.0
        exact_match_accuracy = total_exact_match / num_samples if num_samples > 0 else 0.0
        
        # Log BOTH metrics to W&B
        wandb.log({
            # VQA Accuracy (official metric)
            "eval/vqa_accuracy": vqa_accuracy,
            "eval/vqa_total_score": total_vqa_score,
            
            # Exact Match Accuracy
            "eval/exact_match_accuracy": exact_match_accuracy,
            "eval/exact_match_correct": int(total_exact_match),
            
            # Shared
            "eval/accuracy_num_samples": num_samples,
            "epoch": state.epoch,
        })
        
        # Update bests
        if vqa_accuracy > self.best_vqa_accuracy:
            self.best_vqa_accuracy = vqa_accuracy
            wandb.run.summary["best_vqa_accuracy"] = vqa_accuracy
        
        if exact_match_accuracy > self.best_exact_match:
            self.best_exact_match = exact_match_accuracy
            wandb.run.summary["best_exact_match_accuracy"] = exact_match_accuracy
        
        # Print summary
        print(f"\n{'='*70}")
        print("ACCURACY RESULTS")
        print(f"{'='*70}")
        print(f"VQA Accuracy (Official):     {vqa_accuracy:.2%} (score: {total_vqa_score:.1f}/{num_samples})")
        print(f"Exact Match Accuracy:        {exact_match_accuracy:.2%} ({int(total_exact_match)}/{num_samples})")
        print(f"")
        print(f"Best VQA Accuracy:           {self.best_vqa_accuracy:.2%}")
        print(f"Best Exact Match Accuracy:   {self.best_exact_match:.2%}")
        print(f"{'='*70}\n")
        
        model.train()