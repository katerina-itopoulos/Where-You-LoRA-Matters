# Where You LoRA Matters
Master's thesis investigating LoRA adapter placement and modality collapse in VLMs

# Create virtual environment
python -m venv thesis_env
source thesis_env/bin/activate  # On Windows: thesis_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Login to W&B
wandb login