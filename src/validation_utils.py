# src/validation_utils.py
# src/validation_utils.py

import numpy as np
import torch
import wandb
from transformers import TrainerCallback
import itertools


def generate_validation_configs(lora_ranks, learning_rates, lora_alpha_multiplier):
    """Generate all combinations of hyperparameters for validation."""
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
    Optimized with batched generation.
    """
    
    def __init__(self, processor, eval_dataset, max_new_tokens=16, eval_samples=500, batch_size=4):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.max_new_tokens = max_new_tokens
        self.eval_samples = eval_samples
        self.batch_size = batch_size
        
        self.best_vqa_accuracy = 0.0
        self.best_exact_match = 0.0
    
    def compute_vqa_score(self, prediction, answer_counts):
        pred_norm = prediction.lower().strip()
        for answer, count in answer_counts.items():
            ans_norm = answer.lower().strip()
            if pred_norm == ans_norm:
                return min(count / 3.0, 1.0)
        return 0.0
    
    def compute_exact_match(self, prediction, most_common_answer):
        pred_norm = prediction.lower().strip()
        ans_norm = most_common_answer.lower().strip()
        return 1.0 if pred_norm == ans_norm else 0.0
    
    def collate_batch(self, samples, device):
        import torch

        input_ids = [torch.as_tensor(s['input_ids']) for s in samples]
        attention_mask = [torch.as_tensor(s['attention_mask']) for s in samples]

        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        max_len = max(x.size(0) for x in input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - ids.size(0)

            padded_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_token_id), ids])
            )
            padded_attention_masks.append(
                torch.cat([torch.zeros(pad_len), mask])
            )

        input_ids_padded = torch.stack(padded_input_ids).to(device)
        attention_mask_padded = torch.stack(padded_attention_masks).to(device)

        pixel_values = torch.cat(
            [torch.as_tensor(s['pixel_values']) for s in samples], dim=0
        ).to(device)

        image_grid_thw = torch.stack(
            [torch.as_tensor(s['image_grid_thw']) for s in samples], dim=0
        ).to(device)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    
    def generate_single(self, model, sample):
        """Fallback: generate for a single sample."""
        inputs = {
            'input_ids': sample['input_ids'].unsqueeze(0).to(model.device),
            'attention_mask': sample['attention_mask'].unsqueeze(0).to(model.device),
            'pixel_values': sample['pixel_values'].unsqueeze(0).to(model.device),
            'image_grid_thw': sample['image_grid_thw'].unsqueeze(0).to(model.device),
        }
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        
        start_idx = inputs['input_ids'].shape[1]
        return self.processor.tokenizer.decode(
            generated_ids[0][start_idx:], 
            skip_special_tokens=True
        ).strip()
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        import json
        from tqdm import tqdm
        
        print(f"\n{'='*70}")
        print("🎯 Computing Accuracy Metrics on Validation Set")
        print(f"{'='*70}")

        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        
        model.eval()
        
        total_vqa_score = 0.0
        total_exact_match = 0.0
        num_samples = 0
        
        if self.eval_samples is None:
            eval_size = len(self.eval_dataset)
            indices = list(range(eval_size))
        else:
            eval_size = min(self.eval_samples, len(self.eval_dataset))
            indices = np.random.choice(len(self.eval_dataset), eval_size, replace=False).tolist()
            
        print(f"Evaluating on {eval_size} samples (batch_size={self.batch_size})")
        
        # Collect ground truths
        all_answer_counts = []
        all_most_common = []
        
        for idx in indices:
            sample = self.eval_dataset[int(idx)]
            
            if "answer_counts" in sample:
                answer_counts_raw = sample["answer_counts"]
                if isinstance(answer_counts_raw, str):
                    answer_counts = json.loads(answer_counts_raw)
                else:
                    answer_counts = answer_counts_raw
                most_common_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
            else:
                most_common_answer = sample.get('answer', sample.get('labels', ''))
                answer_counts = {most_common_answer: 10}
            
            all_answer_counts.append(answer_counts)
            all_most_common.append(most_common_answer)
        
        # Process in batches
        all_predictions = []
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, eval_size, self.batch_size), desc="Generating (batched)"):
                batch_end = min(batch_start + self.batch_size, eval_size)
                batch_indices = indices[batch_start:batch_end]
                batch_samples = [self.eval_dataset[int(idx)] for idx in batch_indices]
                
                try:
                    inputs = self.collate_batch(batch_samples, model.device)
                    input_len = inputs['input_ids'].shape[1]
                    
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    
                    for i in range(len(batch_samples)):
                        prediction = self.processor.tokenizer.decode(
                            generated_ids[i][input_len:], 
                            skip_special_tokens=True
                        ).strip()
                        all_predictions.append(prediction)
                
                except Exception as e:
                    print(f"\nBatch failed: {e}")
                    print("Falling back to single inference for this batch...")
                    for sample in batch_samples:
                        try:
                            prediction = self.generate_single(model, sample)
                        except Exception as e2:
                            print(f"Single inference failed: {e2}")
                            prediction = ""
                        all_predictions.append(prediction)
        
        # Compute metrics
        for pred, answer_counts, most_common in zip(all_predictions, all_answer_counts, all_most_common):
            total_vqa_score += self.compute_vqa_score(pred, answer_counts)
            total_exact_match += self.compute_exact_match(pred, most_common)
            num_samples += 1
        
        vqa_accuracy = total_vqa_score / num_samples if num_samples > 0 else 0.0
        exact_match_accuracy = total_exact_match / num_samples if num_samples > 0 else 0.0
        
        wandb.log({
            "eval/vqa_accuracy": vqa_accuracy,
            "eval/vqa_total_score": total_vqa_score,
            "eval/exact_match_accuracy": exact_match_accuracy,
            "eval/exact_match_correct": int(total_exact_match),
            "eval/accuracy_num_samples": num_samples,
            "epoch": state.epoch,
        })
        
        if vqa_accuracy > self.best_vqa_accuracy:
            self.best_vqa_accuracy = vqa_accuracy
            wandb.run.summary["best_vqa_accuracy"] = vqa_accuracy
        
        if exact_match_accuracy > self.best_exact_match:
            self.best_exact_match = exact_match_accuracy
            wandb.run.summary["best_exact_match_accuracy"] = exact_match_accuracy
        
        print(f"\n{'='*70}")
        print("ACCURACY RESULTS")
        print(f"{'='*70}")
        print(f"VQA Accuracy (Official):     {vqa_accuracy:.2%} (score: {total_vqa_score:.1f}/{num_samples})")
        print(f"Exact Match Accuracy:        {exact_match_accuracy:.2%} ({int(total_exact_match)}/{num_samples})")
        print(f"Best VQA Accuracy:           {self.best_vqa_accuracy:.2%}")
        print(f"Best Exact Match Accuracy:   {self.best_exact_match:.2%}")
        print(f"{'='*70}\n")

        #move back to original padding side
        self.processor.tokenizer.padding_side = original_padding_side
        model.train()