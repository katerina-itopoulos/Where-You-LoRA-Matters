"""Utilities for hyperparameter validation experiments"""
import itertools

def generate_validation_configs(lora_ranks, learning_rates, lora_alpha_multiplier):
    """Generate all validation configurations"""
    for rank, lr in itertools.product(lora_ranks, learning_rates):
        yield {
            "lora_r": rank,
            "lora_alpha": rank * lora_alpha_multiplier,
            "learning_rate": lr,
        }

def create_validation_run_name(rank, lr, run_id):
    """Create consistent run names"""
    return f"val_r{rank}_lr{lr:.0e}_id{run_id}"