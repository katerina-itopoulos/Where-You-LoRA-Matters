import numpy as np
import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, TrainerCallback

def initialize_wandb(project_name, run_name, entity, config_dict):
    """Initialize W&B run"""
    print(f"\n{'='*70}")
    print("🚀 Initializing Weights & Biases")
    print(f"{'='*70}")

    wandb.init(
        project=project_name,
        name=run_name,
        entity=entity,
        config=config_dict
    )

    print(f"✓ W&B Run: {wandb.run.name}")
    print(f"✓ URL: {wandb.run.url}")

class WandBLoRAMetricsCallback(TrainerCallback):
    """Logs LoRA-specific metrics to W&B"""

    def __init__(self, compute_freq=5):
        self.compute_freq = compute_freq
        self.best_val_loss = float('inf')

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        """Log metrics to W&B at each logging step"""
        if logs is None:
            return

        # Calculate current epoch
        current_epoch = state.epoch if state.epoch is not None else 0

        # Compute LoRA metrics periodically
        if state.global_step % self.compute_freq == 0:
            lora_metrics = self.compute_lora_metrics(model)
            # Add epoch to metrics for x-axis
            lora_metrics['epoch'] = current_epoch
            wandb.log(lora_metrics)
            self._print_lora_metrics(lora_metrics, current_epoch)

        # Track best model
        if 'eval_loss' in logs:
            if logs['eval_loss'] < self.best_val_loss:
                self.best_val_loss = logs['eval_loss']
                wandb.run.summary["best_val_loss"] = self.best_val_loss
                wandb.run.summary["best_epoch"] = current_epoch

    def compute_lora_metrics(self, model):
        """Compute LoRA-specific metrics"""
        all_eranks = []
        all_stable_ranks = []
        all_frobenius_norms = []
        all_spectral_norms = []

        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                with torch.no_grad():
                    lora_A = module.lora_A['default'].weight.float()
                    lora_B = module.lora_B['default'].weight.float()
                    lora_weight = lora_B @ lora_A

                    # Compute singular values
                    s = torch.linalg.svdvals(lora_weight)

                    # Effective Rank
                    s_normalized = s / (s.sum() + 1e-10)
                    entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum()
                    erank = torch.exp(entropy).item()

                    # Stable Rank
                    frobenius_norm = torch.norm(lora_weight, p='fro').item()
                    spectral_norm = s[0].item()
                    stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2 + 1e-10)

                    all_eranks.append(erank)
                    all_stable_ranks.append(stable_rank)
                    all_frobenius_norms.append(frobenius_norm)
                    all_spectral_norms.append(spectral_norm)

        metrics = {}
        if all_eranks:
            metrics['lora/effective_rank_mean'] = np.mean(all_eranks)
            metrics['lora/effective_rank_std'] = np.std(all_eranks)
            metrics['lora/stable_rank_mean'] = np.mean(all_stable_ranks)
            metrics['lora/frobenius_norm_mean'] = np.mean(all_frobenius_norms)

            # Rank utilization
            nominal_rank = lora_A.shape[0]
            metrics['lora/rank_utilization'] = np.mean(all_eranks) / nominal_rank

        return metrics

    def _print_lora_metrics(self, metrics, epoch):
        """Print LoRA metrics summary"""
        print(f"\n{'='*70}")
        print(f"LoRA Metrics at Epoch {epoch:.2f}")
        print(f"{'='*70}")
        print(f"Effective Rank: {metrics.get('lora/effective_rank_mean', 0):.2f} ± {metrics.get('lora/effective_rank_std', 0):.2f}")
        print(f"Rank Utilization: {metrics.get('lora/rank_utilization', 0)*100:.1f}%")
        print(f"{'='*70}\n")

def upload_to_wandb_artifacts(
    output_dir, run_name, lora_r, lora_alpha,
    model_name, best_val_loss, train_size, val_size
):
    """Upload LoRA checkpoint to W&B Artifacts"""
    print("\n📦 Uploading LoRA checkpoint to W&B Artifacts...")

    artifact = wandb.Artifact(
        name=f"lora-checkpoint-{run_name}",
        type="model",
        description=f"LoRA adapter (rank={lora_r}) for {model_name}",
        metadata={
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "base_model": model_name,
            "best_val_loss": best_val_loss,
            "train_size": train_size,
            "val_size": val_size,
        }
    )

    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    print(f"✓ LoRA checkpoint uploaded to W&B!")
    return artifact

def download_lora_from_wandb(
    artifact_name,
    base_model_name,
    wandb_entity=None,
    merge=False ):
    """
    Load a LoRA checkpoint from W&B Artifacts

    Args:
        artifact_name: Name of the artifact (e.g., "lora-checkpoint-quick-test-r4:latest")
        base_model_name: Base model to load
        wandb_entity: Your W&B username/team
        merge: If True, merge LoRA into base model for faster inference

    Returns:
        model: The model with LoRA adapter loaded (or merged)
    """
    api = wandb.Api()

    if wandb_entity:
        artifact_path = f"{wandb_entity}/{artifact_name}"
    else:
        artifact_path = artifact_name

    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    print(f"✓ Downloaded to: {artifact_dir}")


def load_lora(base_model_name, artifact_dir, merge):

  """Load base model"""
  print(f"\n Loading base model: {base_model_name}")

  base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

  # Load LoRA adapter
  print(f"\n🔧 Loading LoRA adapter...")
  model = PeftModel.from_pretrained(base_model, artifact_dir)

  print(f"✓ LoRA adapter loaded!")
  print(f"  Metadata: {artifact.metadata}")

  # Optionally merge
  if merge:
      print(f"\n🔀 Merging LoRA into base model...")
      model = model.merge_and_unload()
      print(f"✓ Merged! Model is now standalone.")

  print(f"\n{'='*70}")
  print("✅ Model ready for inference!")
  print(f"{'='*70}\n")

  return model

