import os

import torch
import wandb
from peft import LoraConfig, TaskType
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .data_preprocessing import VLDataCollatorPadTorch
from .validation_utils import VQAAccuracyCallback
from .wandb_utils import WandBLoRAMetricsCallback


def create_lora_config_vl(
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    target_modules: list[str],
) -> LoraConfig:
    """Return a LoRA config for Qwen3-VL with uniform rank across all target modules."""
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
    lora_dropout: float,
    target_modules: list[str],
    rank_pattern: dict[str, int] | None = None,
    alpha_pattern: dict[str, float] | None = None,
) -> LoraConfig:
    """
    Return a LoRA config with per-module rank and alpha overrides.

    Args:
        lora_dropout: Dropout probability applied to LoRA layers.
        target_modules: List of module name patterns to apply LoRA to.
        rank_pattern: Regex-keyed dict mapping full module names to ranks.
        alpha_pattern: Regex-keyed dict mapping full module names to alpha values.
    """
    return LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        rank_pattern=rank_pattern or {},
        alpha_pattern=alpha_pattern or {},
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )


def setup_optimizer_scheduler(
    learning_rate: float = 5e-5,
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
) -> dict:
    """Return a dict of optimizer and scheduler kwargs for use in TrainingArguments."""
    return {
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
    }


def create_optimizer_with_param_groups(
    model: torch.nn.Module,
    vision_lr: float = 5e-5,
    llm_lr: float = 5e-5,
    projector_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Return an AdamW optimizer with separate learning rates for vision, projector, and LLM LoRA params."""
    vision_params, llm_params, projector_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual.blocks" in name:
            vision_params.append(param)
        elif "visual.merger" in name or "visual.deepstack" in name:
            projector_params.append(param)
        else:
            llm_params.append(param)

    print(f"Vision params:     {len(vision_params)} tensors, lr={vision_lr}")
    print(f"LLM params:        {len(llm_params)} tensors, lr={llm_lr}")
    print(f"Projector params:  {len(projector_params)} tensors, lr={projector_lr}")

    param_groups = []
    if vision_params:
        param_groups.append({"params": vision_params, "lr": vision_lr, "weight_decay": weight_decay})
    if llm_params:
        param_groups.append({"params": llm_params, "lr": llm_lr, "weight_decay": weight_decay})
    if projector_params:
        param_groups.append({"params": projector_params, "lr": projector_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(param_groups)


def train_vl_lora_with_wandb(
    model: torch.nn.Module,
    processor,
    train_dataset,
    val_dataset,
    val_accuracy_dataset,
    test_dataset,
    max_grad_norm: float,
    wandb_project: str = "lora-vqav2",
    wandb_run_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_config: dict | None = None,
    output_dir: str = "./lora-vqav2-output",
    epochs: int = 3,
    max_steps: int = -1,
    logging_steps: int = 5,
    eval_strategy: str = "epoch",
    eval_steps: int | None = None,
    save_strategy: str | None = None,
    save_steps: int | None = None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    optimizer_config: dict | None = None,
    use_separate_lrs: bool = False,
    vision_lr: float = 5e-5,
    llm_lr: float = 5e-5,
    projector_lr: float = 1e-4,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.01,
    compute_vqa_accuracy: bool = True,
    vqa_eval_samples: int = 500,
    vqa_batch_size: int = 8,
) -> tuple[Trainer, dict]:
    """
    Train a Qwen3-VL LoRA model with W&B logging, optional per-module LRs, and VQA accuracy evaluation.

    Returns:
        Tuple of ``(trainer, results)`` where ``results`` contains best val loss,
        VQA accuracy, and exact match accuracy.
    """
    if optimizer_config is None:
        optimizer_config = setup_optimizer_scheduler()

    if save_strategy is None:
        save_strategy = eval_strategy

    if save_steps is None:
        save_steps = eval_steps

    if eval_strategy == "steps" and eval_steps is None:
        raise ValueError("eval_steps must be provided when eval_strategy='steps'")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    full_config = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "val_accuracy_size": len(val_accuracy_dataset) if val_accuracy_dataset else 0,
        "test_size": len(test_dataset) if test_dataset else 0,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": 100 * trainable_params / total_params,
        "max_steps": max_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "compute_vqa_accuracy": compute_vqa_accuracy,
        "vqa_eval_samples": vqa_eval_samples,
        "use_separate_lrs": use_separate_lrs,
        "vqa_batch_size": vqa_batch_size,
        **optimizer_config,
        **(wandb_config or {}),
    }

    if use_separate_lrs:
        full_config.update({"vision_lr": vision_lr, "llm_lr": llm_lr, "projector_lr": projector_lr})

    wandb.init(project=wandb_project, name=wandb_run_name, entity=wandb_entity, config=full_config)
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    print(f"W&B Run: {wandb.run.name}")
    print(f"URL: {wandb.run.url}")

    optimizer = None
    if use_separate_lrs:
        optimizer = create_optimizer_with_param_groups(
            model=model,
            vision_lr=vision_lr,
            llm_lr=llm_lr,
            projector_lr=projector_lr,
            weight_decay=optimizer_config.get("weight_decay", 0.01),
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=wandb_run_name,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=optimizer_config["learning_rate"] if not use_separate_lrs else 1e-5,
        lr_scheduler_type=optimizer_config["lr_scheduler_type"],
        warmup_steps=optimizer_config["warmup_steps"],
        weight_decay=optimizer_config.get("weight_decay", 0.01),
        optim="adamw_torch",
        max_grad_norm=max_grad_norm,
        fp16=False,
        bf16=True,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_on_start=True,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        report_to="wandb",
        remove_unused_columns=False,
    )

    steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
    print(f"\n{'='*70}")
    print("Training Schedule")
    print(f"{'='*70}")
    print(f"Steps per epoch:  {steps_per_epoch}")
    print(f"Total epochs:     {epochs}")
    print(f"Total steps:      {steps_per_epoch * epochs}")
    print(f"Eval strategy:    {eval_strategy}")
    print(f"VQA Accuracy:     {'enabled' if compute_vqa_accuracy else 'disabled'}")
    if eval_strategy == "steps":
        print(f"Eval every:       {eval_steps} steps (~{(steps_per_epoch * epochs) // eval_steps} total)")
    else:
        print(f"Eval every:       1 epoch ({steps_per_epoch} steps, {epochs} total)")
    print(f"{'='*70}\n")

    lora_metrics_callback = WandBLoRAMetricsCallback(compute_freq=logging_steps)
    callbacks = [lora_metrics_callback]

    vqa_callback = None
    if compute_vqa_accuracy:
        vqa_callback = VQAAccuracyCallback(
            processor=processor,
            eval_dataset=val_accuracy_dataset,
            max_new_tokens=16,
            eval_samples=vqa_eval_samples,
            batch_size=vqa_batch_size,
        )
        callbacks.append(vqa_callback)

    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VLDataCollatorPadTorch(),
        callbacks=callbacks,
        optimizers=(optimizer, None) if optimizer else (None, None),
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    test_dataset.save_to_disk(os.path.join(output_dir, "test_dataset"))

    wandb.run.summary["final_train_loss"] = trainer.state.log_history[-1].get("loss")
    wandb.run.summary["best_val_loss"] = lora_metrics_callback.best_val_loss

    if vqa_callback is not None:
        wandb.run.summary["best_vqa_accuracy"] = vqa_callback.best_vqa_accuracy
        wandb.run.summary["best_exact_match_accuracy"] = vqa_callback.best_exact_match

    try:
        artifact = wandb.Artifact(
            name=f"lora-checkpoint-{wandb.run.id}",
            type="model",
            description="LoRA adapter checkpoint",
            metadata={
                "best_val_loss": lora_metrics_callback.best_val_loss,
                "best_vqa_accuracy": vqa_callback.best_vqa_accuracy if vqa_callback else None,
                "best_exact_match_accuracy": vqa_callback.best_exact_match if vqa_callback else None,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
            },
        )
        artifact.add_dir(output_dir)
        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait()
        print(f"Artifact uploaded: {artifact.name}")
    except Exception as e:
        print(f"Artifact upload failed: {e}")
        print("Model saved locally, continuing...")

    print(f"\n{'='*70}")
    print("Training complete")
    print(f"Local model: {output_dir}")
    print(f"Dashboard:   {wandb.run.url}")
    if vqa_callback is not None:
        print(f"Best VQA Accuracy: {vqa_callback.best_vqa_accuracy:.2%}")
        print(f"Best Exact Match:  {vqa_callback.best_exact_match:.2%}")
    print(f"{'='*70}\n")

    results = {
        "best_val_loss": lora_metrics_callback.best_val_loss,
        "best_vqa_accuracy": vqa_callback.best_vqa_accuracy if vqa_callback else None,
        "best_exact_match": vqa_callback.best_exact_match if vqa_callback else None,
    }

    wandb.finish()
    return trainer, results