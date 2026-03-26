# Where You LoRA Matters

Master's thesis investigating how LoRA adapter placement affects modality collapse in Vision-Language Models (VLMs).

## Overview

This repository contains the code for fine-tuning and evaluating **Qwen3-VL-8B** under six LoRA adapter configurations to study how adapter placement influences representational collapse across visual and language modalities.

### LoRA Configurations

| Configuration | Adapted Modules | 
|---|---|---|
| **LLM-only** | Language model attention layers | 
| **Projector-only** | Vision-language projector | 
| **Vision + Projector** | Vision encoder + projector | 
| **LLM + Projector** | Language model + projector | 
| **Vision + LLM** | Vision encoder + language model | 
| **Vision + LLM + Projector** | Vision encoder + language model + projector | 

### Collapse Metrics

Modality collapse is measured using:
- **CKA** (Centered Kernel Alignment) — primary indicator of representational similarity pre/post fine-tuning
- **Normalized Effective Rank** — dimensionality of learned representations
- **Concentration Ratio (CR)** — variance concentration in top singular values
- **Intra-modal Similarity** — within-modality representational uniformity

### Benchmarks

Models are evaluated on:
- **VQAv2** — visual question answering (also used as training data, 20K samples)
- **HallusionBench** — hallucination detection
- **MMVP** — multimodal visual perception
- **DASH-B** — balanced diagnostic benchmark
- **VQA-VS** — out-of-distribution VQA evaluation

## Setup

### Requirements

- Python 3.10+
- CUDA-compatible GPU (experiments run on AWS EC2/SageMaker g5 2XL instance)
- [Weights & Biases](https://wandb.ai/) account for experiment tracking

### Installation

```bash
# Clone the repository
git clone https://github.com/katerina-itopoulos/Where-You-LoRA-Matters.git
cd Where-You-LoRA-Matters

# Create virtual environment or conda env
python -m venv thesis_env
source thesis_env/bin/activate  # On Windows: thesis_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Login to Weights & Biases
wandb login
```

## Repository Structure

```
Where-You-LoRA-Matters/
├── scripts/
│   ├── finetune.py                 # Fine-tuning for single-component configs (uniform rank/LR)
│   ├── finetune_multi.py           # Fine-tuning for multi-component configs (per-module rank/LR)
│   ├── evaluate.py                 # Benchmark evaluation across all test sets
│   ├── process_datasets_vqa.py     # Preprocess VQAv2 dataset
│   ├── process_dataset_vqa_vs.py   # Preprocess VQA-VS dataset
│   ├── qualitative_analysis.py     # Generate HTML reports with qualitative results
│   ├── visualize_results_wandb.py  # Push aggregated downstream and collapse metrics to W&B
├── src/
│   ├── collapse_metrics.py         # CKA, effective rank, CR, intra-modal similarity
│   ├── data_preprocessing.py       # Data loading and preprocessing utilities
│   ├── datasets.py                 # Dataset classes and formatting
│   ├── experiments.py              # Experiment configuration management
│   ├── finetuning_utils.py         # LoRA config creation, optimizer setup, training loops
│   ├── inference_utils.py          # Inference and generation utilities
│   ├── model_utils.py              # Model loading and LoRA setup
│   ├── validation_utils.py         # Validation and accuracy computation
│   ├── vqa_metrics.py              # VQA accuracy metrics
│   └── wandb_utils.py              # W&B logging utilities
├── requirements.txt
├── LICENSE
└── README.md
```

## Usage

### 1. Data Preprocessing

Preprocess the VQAv2 and VQA-VS datasets into the expected parquet format:

```bash
python scripts/process_datasets_vqa.py
python scripts/process_dataset_vqa_vs.py
```

### 2. Fine-tuning

There are two fine-tuning scripts depending on the LoRA configuration:

**`finetune.py`** — for single-component configurations where one uniform LoRA rank and learning rate applies to all target modules (LLM-only, Projector-only). Set `PLACEMENT_STRATEGY`, `TARGET_MODULES`, and `LEARNING_RATE` at the top of the script:

```bash
python scripts/finetune.py
```

**`finetune_multi.py`** — for multi-component configurations that require per-module rank patterns and separate learning rates (Vision+LLM, Vision+Projector, LLM+Projector, Vision+LLM+Projector). Set `PLACEMENT_STRATEGY`, target module combination, and per-component ranks/LRs at the top:

```bash
python scripts/finetune_multi.py
```

Key hyperparameters (configured in scripts):
- **LoRA rank**: 64
- **Training data**: 20K VQAv2 samples
- **Epochs**: 2.4
- **Batch size**: 2 (with gradient accumulation steps of 4)
- **Learning rates**: 1e-4 (LLM/projector), 1e-5 (vision encoder)
- **Scheduler**: Cosine with 500 warmup steps

### 3. Evaluation

Run benchmark evaluation on a fine-tuned checkpoint:

```bash
python scripts/evaluate.py
```

### 4. Analysis

Generate qualitative HTML reports and push collapse metrics to W&B:

```bash
python scripts/qualitative_analysis.py
python scripts/visualize_results_wandb.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@mastersthesis{where_you_lora_matters,
  title={Where You LoRA Matters: Investigating Adapter Placement and Modality Collapse in Vision-Language Models},
  author={Katerina Itopoulos},
  year={2026},
  school={SRH-Berlin}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
