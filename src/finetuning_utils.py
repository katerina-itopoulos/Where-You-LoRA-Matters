import os
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
import torch

from .data_preprocessing import VLDataCollatorPadTorch
from .wandb_utils import WandBLoRAMetricsCallback
from .validation_utils import VQAAccuracyCallback


def create_lora_config_vl(lora_r, lora_alpha, lora_dropout, target_modules):
    """
    Create LoRA configuration for Qwen3-VL
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

def create_lora_config_vl_with_rank_pattern(
    lora_dropout,
    target_modules,
    rank_pattern=None,  # Dict mapping regex patterns to ranks
    alpha_pattern=None,
):
    """
    Create LoRA configuration with different ranks per module pattern
    
    Example:
        rank_pattern = {
            "visual.merger": 64,
            "visual.blocks": 32,
            "model.model": 32,
        }
    """
    # Default rank for modules not matching any pattern
    default_rank = 64
    default_alpha = 128
    
    return LoraConfig(
        r=default_rank,
        lora_alpha=default_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        rank_pattern=rank_pattern or {},
        alpha_pattern=alpha_pattern or {},
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )

def setup_optimizer_scheduler(
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    weight_decay=0.01,
):
    """
    Configure optimizer and learning rate scheduler
    """
    return {
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
    }


def create_optimizer_with_param_groups(
    model,
    llm_lr=5e-5,
    projector_lr=1e-4,
    weight_decay=0.01,
):
    """
    Create optimizer with different learning rates for LLM vs projector LoRA adapters
    """
    llm_params = []
    projector_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if parameter belongs to visual/projector modules
        if "visual" in name or "projector" in name:
            projector_params.append(param)
        else:
            llm_params.append(param)
    
    print(f"LLM params: {len(llm_params)}, Projector params: {len(projector_params)}")
    
    optimizer = torch.optim.AdamW([
        {"params": llm_params, "lr": llm_lr, "weight_decay": weight_decay},
        {"params": projector_params, "lr": projector_lr, "weight_decay": weight_decay},
    ])
    return optimizer

def train_vl_lora_with_wandb(
    # Required inputs
    model,
    processor,
    train_dataset,
    val_dataset,
    val_accuracy_dataset,
    test_dataset,
    max_grad_norm,
    
    # W&B config
    wandb_project="lora-vqav2",
    wandb_run_name=None,
    wandb_entity=None,
    wandb_config=None,

    # Output
    output_dir="./lora-vqav2-output",

    # Training schedule
    epochs=3,
    max_steps=-1,
    logging_steps=5,
    
    # Evaluation strategy
    eval_strategy="epoch",
    eval_steps=None,
    save_strategy=None,
    save_steps=None,

    # Batch size
    batch_size=1,
    gradient_accumulation_steps=4,

    # Optimization
    optimizer_config=None,
    use_separate_lrs=False,
    llm_lr=5e-5,
    projector_lr=1e-4,

    # Early stopping
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
    
    # VQA accuracy evaluation
    compute_vqa_accuracy=True,
    vqa_eval_samples=500,
):
    """
    Main training function with optional VQA accuracy evaluation and separate LRs.
    """

    if optimizer_config is None:
        optimizer_config = setup_optimizer_scheduler()

    if wandb_config is None:
        wandb_config = {}
    
    if save_strategy is None:
        save_strategy = eval_strategy
    
    if save_steps is None:
        save_steps = eval_steps
    
    if eval_strategy == "steps" and eval_steps is None:
        raise ValueError("eval_steps must be provided when eval_strategy='steps'")

    # 1. Initialize W&B
    print(f"\n{'='*70}")
    print("🚀 Initializing Weights & Biases")
    print(f"{'='*70}")

    full_config = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "val_accuracy_size": len(val_accuracy_dataset) if val_accuracy_dataset else 0,
        "test_size": len(test_dataset) if test_dataset else 0,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "max_steps": max_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "compute_vqa_accuracy": compute_vqa_accuracy,
        "vqa_eval_samples": vqa_eval_samples,
        "use_separate_lrs": use_separate_lrs,
        **optimizer_config,
        **wandb_config,
    }
    
    if use_separate_lrs:
        full_config["llm_lr"] = llm_lr
        full_config["projector_lr"] = projector_lr
    
    full_config["trainable_percent"] = 100 * full_config["trainable_params"] / full_config["total_params"]

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        entity=wandb_entity,
        config=full_config
    )

    print(f"✓ W&B Run: {wandb.run.name}")
    print(f"✓ URL: {wandb.run.url}")

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # 2. Create custom optimizer if using separate LRs
    optimizer = None
    if use_separate_lrs:
        optimizer = create_optimizer_with_param_groups(
            model=model,
            llm_lr=llm_lr,
            projector_lr=projector_lr,
            weight_decay=optimizer_config.get("weight_decay", 0.01)
        )
        print(f"✓ Using separate LRs: LLM={llm_lr}, Projector={projector_lr}")

    # 3. Create training arguments
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

        # Optimization - only set learning_rate if NOT using separate LRs
        learning_rate=optimizer_config["learning_rate"] if not use_separate_lrs else None,
        lr_scheduler_type=optimizer_config["lr_scheduler_type"],
        warmup_steps=optimizer_config["warmup_steps"],
        weight_decay=optimizer_config.get("weight_decay", 0.01),
        optim="adamw_torch",
        max_grad_norm=max_grad_norm,

        # Precision
        fp16=False,
        bf16=True,

        # Evaluation & logging
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_on_start=True,

        # Checkpointing
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Memory optimization
        gradient_checkpointing=True,

        # Data loading optimization
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # W&B
        report_to="wandb",

        # VL-specific
        remove_unused_columns=False,
    )
    
    # Print training schedule info
    print(f"\n{'='*70}")
    print("📊 Training Schedule")
    print(f"{'='*70}")
    steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total epochs: {epochs}")
    print(f"Total steps: {steps_per_epoch * epochs}")
    print(f"Eval strategy: {eval_strategy}")
    print(f"VQA Accuracy: {'Enabled' if compute_vqa_accuracy else 'Disabled'}")

    if eval_strategy == "steps":
        print(f"Eval every: {eval_steps} steps")
        print(f"Total evaluations: ~{(steps_per_epoch * epochs) // eval_steps}")
    else:
        print(f"Eval every: 1 epoch ({steps_per_epoch} steps)")
        print(f"Total evaluations: {epochs}")
    print(f"{'='*70}\n")

    # 4. Initialize callbacks
    callbacks = []
    
    # LoRA metrics
    lora_metrics_callback = WandBLoRAMetricsCallback(compute_freq=logging_steps)
    callbacks.append(lora_metrics_callback)
    
    # VQA accuracy callback
    if compute_vqa_accuracy:
        vqa_callback = VQAAccuracyCallback(
            processor=processor,
            eval_dataset=val_accuracy_dataset,
            max_new_tokens=16,
            eval_samples=vqa_eval_samples,
        )
        callbacks.append(vqa_callback)
    
    # Early stopping
    if early_stopping_patience is not None and early_stopping_patience > 0:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )
        callbacks.append(early_stopping)

    # 5. Create data collator
    data_collator = VLDataCollatorPadTorch()

    # 6. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=(optimizer, None) if optimizer else (None, None),
    )
    
    print("Starting training...")
    trainer.train()
    
    print("\nSaving model...")
    trainer.save_model(output_dir)
    test_dataset.save_to_disk(os.path.join(output_dir, "test_dataset"))

    # Update W&B summary
    wandb.run.summary["final_train_loss"] = trainer.state.log_history[-1].get("loss", None)
    wandb.run.summary["best_val_loss"] = lora_metrics_callback.best_val_loss
    
    # Add VQA accuracy to summary
    if compute_vqa_accuracy:
        wandb.run.summary["best_vqa_accuracy"] = vqa_callback.best_vqa_accuracy
        wandb.run.summary["best_exact_match_accuracy"] = vqa_callback.best_exact_match

    # Update artifact metadata
    artifact = wandb.Artifact(
        name=f"lora-checkpoint-{wandb.run.id}",
        type="model",
        description=f"LoRA adapter checkpoint",
        metadata={
            "best_val_loss": lora_metrics_callback.best_val_loss,
            "best_vqa_accuracy": vqa_callback.best_vqa_accuracy if compute_vqa_accuracy else None,
            "best_exact_match_accuracy": vqa_callback.best_exact_match if compute_vqa_accuracy else None,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        }
    )

    artifact.add_dir(output_dir)
    logged_artifact = wandb.log_artifact(artifact)
    logged_artifact.wait()

    print(f"✓ Artifact uploaded: {artifact.name}")

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Local model: {output_dir}")
    print(f"W&B Artifact: {artifact.name}")
    print(f"Dashboard: {wandb.run.url}")
    
    # Print final accuracy
    if compute_vqa_accuracy:
        print(f"Best VQA Accuracy: {vqa_callback.best_vqa_accuracy:.2%}")
        print(f"Best Exact Match: {vqa_callback.best_exact_match:.2%}")

    results = {
        "best_val_loss": lora_metrics_callback.best_val_loss,
        "best_vqa_accuracy": vqa_callback.best_vqa_accuracy if compute_vqa_accuracy else None,
        "best_exact_match": vqa_callback.best_exact_match if compute_vqa_accuracy else None,
    }

    wandb.finish()
    
    return trainer, results