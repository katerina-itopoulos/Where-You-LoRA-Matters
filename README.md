# Where You LoRA Matters: Adapter Placement and Modality Collapse in VLMs

This repository contains the implementation and experiments for investigating how LoRA adapter placement affects modality collapse and task performance in Vision-Language Models (VLMs). The research compares strategic adapter placements across vision-only, language-only, joint, and comprehensive configurations.


## Research Questions

- How does adapter placement (vision-only, language-only, joint, comprehensive) affect modality balance, task performance, and parameter efficiency?
- What is the optimal placement strategy for balancing performance requirements, computational constraints, and multimodal robustness?

## Repository Structure
├── src/
│   ├── models/          # Model implementations and adapter configurations
│   ├── data/            # Dataset loading and preprocessing
│   ├── metrics/         # Modality collapse and performance metrics
│   ├── training/        # Fine-tuning scripts and configurations
│   └── evaluation/      # Evaluation and analysis scripts
├── configs/             # Configuration files for different experiments
├── scripts/             # Bash scripts for running experiments
├── notebooks/           # Jupyter notebooks for analysis and visualization
├── results/             # Experimental results and outputs
├── docs/                # Documentation and thesis materials
└── requirements.txt     # Python dependencies

## Contact
Katerina Itopoulos - katerinaitopoulos@icloud.com
[Project Link](https://github.com/katerina-itopoulos/Where-You-LoRA-Matters)

This research is conducted as part of a Master's thesis investigating modality collapse in vision-language models.