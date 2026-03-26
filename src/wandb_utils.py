import numpy as np
import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from wandb.sdk.wandb_run import Run
from wandb.sdk.artifacts.artifact import Artifact


def initialize_wandb(
    project_name: str,
    run_name: str,
    entity: str,
    config_dict: dict,
) -> Run:
    """Initialize a W&B run and return the run object."""
    print(f"\n{'='*70}")
    print("Initializing Weights & Biases")
    print(f"{'='*70}")

    run = wandb.init(
        project=project_name,
        name=run_name,
        entity=entity,
        config=config_dict,
    )

    print(f"W&B Run: {wandb.run.name}")
    print(f"URL: {wandb.run.url}")

    return run


class WandBLoRAMetricsCallback(TrainerCallback):
    """Trainer callback that logs LoRA-specific metrics to W&B."""

    def __init__(self, compute_freq: int = 5) -> None:
        self.compute_freq = compute_freq
        self.best_val_loss: float = float("inf")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        """Log LoRA metrics to W&B at each logging step."""
        if logs is None:
            return

        current_epoch: float = state.epoch if state.epoch is not None else 0.0

        if state.global_step % self.compute_freq == 0:
            lora_metrics = self.compute_lora_metrics(model)
            lora_metrics["epoch"] = current_epoch
            wandb.log(lora_metrics)
            self._print_lora_metrics(lora_metrics, current_epoch)

        if "eval_loss" in logs and logs["eval_loss"] < self.best_val_loss:
            self.best_val_loss = logs["eval_loss"]
            wandb.run.summary["best_val_loss"] = self.best_val_loss
            wandb.run.summary["best_epoch"] = current_epoch

    def compute_lora_metrics(self, model: torch.nn.Module) -> dict[str, float]:
        """Compute effective rank, stable rank, and norm stats across all LoRA layers."""
        all_eranks: list[float] = []
        all_stable_ranks: list[float] = []
        all_frobenius_norms: list[float] = []

        nominal_rank: int = 0

        for _, module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue

            with torch.no_grad():
                lora_A: torch.Tensor = module.lora_A["default"].weight.float()
                lora_B: torch.Tensor = module.lora_B["default"].weight.float()
                lora_weight: torch.Tensor = lora_B @ lora_A

                s: torch.Tensor = torch.linalg.svdvals(lora_weight)

                s_norm = s / (s.sum() + 1e-10)
                entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
                erank: float = torch.exp(entropy).item()

                frob: float = torch.norm(lora_weight, p="fro").item()
                spectral: float = s[0].item()
                stable_rank: float = (frob**2) / (spectral**2 + 1e-10)

                all_eranks.append(erank)
                all_stable_ranks.append(stable_rank)
                all_frobenius_norms.append(frob)
                nominal_rank = lora_A.shape[0]

        if not all_eranks:
            return {}

        mean_erank = float(np.mean(all_eranks))
        return {
            "lora/effective_rank_mean": mean_erank,
            "lora/effective_rank_std": float(np.std(all_eranks)),
            "lora/stable_rank_mean": float(np.mean(all_stable_ranks)),
            "lora/frobenius_norm_mean": float(np.mean(all_frobenius_norms)),
            "lora/rank_utilization": mean_erank / nominal_rank if nominal_rank else 0.0,
        }

    def _print_lora_metrics(self, metrics: dict[str, float], epoch: float) -> None:
        """Print a brief LoRA metrics summary to stdout."""
        print(f"\n{'='*70}")
        print(f"LoRA Metrics at Epoch {epoch:.2f}")
        print(f"{'='*70}")
        print(
            f"Effective Rank: {metrics.get('lora/effective_rank_mean', 0):.2f}"
            f" ± {metrics.get('lora/effective_rank_std', 0):.2f}"
        )
        print(f"Rank Utilization: {metrics.get('lora/rank_utilization', 0) * 100:.1f}%")
        print(f"{'='*70}\n")


def upload_to_wandb_artifacts(
    output_dir: str,
    run_name: str,
    lora_r: int,
    lora_alpha: float,
    model_name: str,
    best_val_loss: float,
    train_size: int,
    val_size: int,
) -> Artifact:
    """Upload a LoRA checkpoint directory to W&B Artifacts and return the artifact."""
    print("\nUploading LoRA checkpoint to W&B Artifacts...")

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
        },
    )

    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    print("LoRA checkpoint uploaded to W&B.")
    return artifact


def download_lora_from_wandb(
    artifact_name: str,
    wandb_entity: str | None = None,
) -> str:
    """
    Download a LoRA artifact from W&B and return the local directory path.

    Args:
        artifact_name: Artifact name, e.g. ``"lora-checkpoint-run:latest"``.
        wandb_entity: W&B username or team name. Prepended to the artifact path when provided.
    """
    api = wandb.Api()
    artifact_path = f"{wandb_entity}/{artifact_name}" if wandb_entity else artifact_name
    artifact = api.artifact(artifact_path)
    artifact_dir: str = artifact.download()

    print(f"Downloaded to: {artifact_dir}")
    return artifact_dir


def load_lora(
    base_model_name: str,
    artifact_dir: str,
    merge: bool = False,
) -> torch.nn.Module:
    """
    Load a base model and attach (or merge) a LoRA adapter from a local directory.

    Args:
        base_model_name: HuggingFace model ID for the base model.
        artifact_dir: Local path containing the LoRA adapter weights.
        merge: When ``True``, merges the adapter into the base model weights
               and unloads it for faster inference.
    """
    print(f"\nLoading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("\nLoading LoRA adapter...")
    model: torch.nn.Module = PeftModel.from_pretrained(base_model, artifact_dir)
    print("LoRA adapter loaded!")

    if merge:
        print("\nMerging LoRA into base model...")
        model = model.merge_and_unload()
        print("Merged. Model is now standalone.")

    print(f"\n{'='*70}")
    print("Model ready for inference!")
    print(f"{'='*70}\n")

    return model