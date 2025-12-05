import os

import wandb
from peft import LoraConfig, TaskType
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .data_preprocessing import VLDataCollator
from .wandb_utils import WandBLoRAMetricsCallback


def create_lora_config_vl(lora_r, lora_alpha, lora_dropout, target_modules):
    """
    Create LoRA configuration for Qwen2.5-VL

    Target modules for Qwen2.5-VL language model part:
    - q_proj, k_proj, v_proj, o_proj (attention)
    - gate_proj, up_proj, down_proj (FFN)

    Note: We typically don't apply LoRA to the vision encoder
    """
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )

def setup_optimizer_scheduler(
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    weight_decay=0.01,
):
    """
    Configure optimizer and learning rate scheduler

    Args:
        learning_rate: Peak learning rate
        lr_scheduler_type: Type of LR scheduler (cosine, linear, constant, etc.)
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay coefficient

    Returns:
        dict: Configuration for optimizer and scheduler
    """
    return {
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
    }

def train_vl_lora_with_wandb(
    # Required inputs
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    processor,

    # W&B config
    wandb_project="lora-vqav2",
    wandb_run_name=None,
    wandb_entity=None,
    wandb_config=None,  # Additional config to log to W&B

    # Output
    output_dir="./lora-vqav2-output",

    # Training schedule
    epochs=3,
    max_steps=50,
    eval_steps=10,
    logging_steps=5,
    save_steps=None,  # Default: same as eval_steps

    # Batch size
    batch_size=1,
    gradient_accumulation_steps=4,

    # Optimization (pass dict from setup_optimizer_scheduler)
    optimizer_config=None,

    # Early stopping
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
):
    """
    Main training function - pure orchestration of training loop

    Args:
        model: PEFT model with LoRA already applied
        train_dataset: Prepared training dataset
        val_dataset: Prepared validation dataset
        test_dataset: Prepared test dataset
        processor: Processor for collation
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        wandb_entity: W&B entity (username/team)
        wandb_config: Additional config dict to log to W&B
        output_dir: Where to save checkpoints
        max_steps: Maximum training steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps (default: same as eval_steps)
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        optimizer_config: Dict from setup_optimizer_scheduler()
        early_stopping_patience: Patience for early stopping
        early_stopping_threshold: Threshold for early stopping

    Returns:
        trainer: Trained Trainer object
    """

    # Set defaults
    if save_steps is None:
        save_steps = eval_steps

    if optimizer_config is None:
        optimizer_config = setup_optimizer_scheduler()

    if wandb_config is None:
        wandb_config = {}

    # 1. Initialize W&B
    print(f"\n{'='*70}")
    print("🚀 Initializing Weights & Biases")
    print(f"{'='*70}")

    # Build W&B config
    full_config = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        **optimizer_config,
        **wandb_config,  # User-provided config
    }
    full_config["trainable_percent"] = 100 * full_config["trainable_params"] / full_config["total_params"]

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        entity=wandb_entity,
        config=full_config
    )

    print(f"✓ W&B Run: {wandb.run.name}")
    print(f"✓ URL: {wandb.run.url}")

    # Define epoch as the x-axis for all metrics
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # 2. Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=wandb_run_name,

        # Training schedule
        num_train_epochs=epochs,
        max_steps=max_steps,

        # Batch size
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # Optimization
        learning_rate=optimizer_config["learning_rate"],
        lr_scheduler_type=optimizer_config["lr_scheduler_type"],
        warmup_steps=optimizer_config["warmup_steps"],
        weight_decay=optimizer_config.get("weight_decay", 0.01),
        optim="adamw_torch",

        # Precision
        bf16=True,

        # Evaluation & logging
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        logging_first_step=True,

        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Memory optimization
        gradient_checkpointing=True,

        # W&B
        report_to="wandb",

        # VL-specific
        remove_unused_columns=False,
    )

    # 3. Initialize callbacks
    lora_metrics_callback = WandBLoRAMetricsCallback(compute_freq=logging_steps)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold
    )

    # 4. Create data collator
    data_collator = VLDataCollator(processor)

    # 5. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[lora_metrics_callback, early_stopping],
    )
    print("Starting training...")

    trainer.train()
    print("\nSaving model...")
    trainer.save_model(output_dir)
    test_dataset.save_to_disk(os.path.join(output_dir, "test_dataset"))

    wandb.run.summary["final_train_loss"] = trainer.state.log_history[-1].get("loss", None)
    wandb.run.summary["best_val_loss"] = lora_metrics_callback.best_val_loss

    artifact = wandb.Artifact(
        name=f"lora-checkpoint-{wandb.run.name}",
        type="model",
        description=f"LoRA adapter checkpoint",
        metadata={
            "best_val_loss": lora_metrics_callback.best_val_loss,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        }
    )

    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    print(f"✓ Artifact uploaded: {artifact.name}")

    # 10. Print summary
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Local model: {output_dir}")
    print(f"W&B Artifact: {artifact.name}")
    print(f"Dashboard: {wandb.run.url}")
    wandb.finish()
    return trainer
