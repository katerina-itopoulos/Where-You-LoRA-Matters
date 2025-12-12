# On your GCP instance, install dependencies:
#sudo apt update
#sudo apt install -y python3-venv python3-pip tmux
#python3 -m venv thesis_env
#source thesis_env/bin/activate
#pip install transformers==4.57.3 datasets qwen-vl-utils gcsfs accelerate pillow torchvision

# Authenticate with GCS
#gcloud auth application-default login

from transformers import AutoProcessor
from src.datasets import prepare_vqav2_datasets_preprocessed_ultra_lean

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MIN_PIXELS = 128*32*32
MAX_PIXELS = 256*32*32

# Dataset sizes to preprocess
TRAIN_SIZE = 5000
VAL_SIZE = 500
TEST_SIZE = 500

GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/preprocessed_vqa"

if __name__ == "__main__":
    print("="*70)
    print("PREPROCESSING VQAv2 DATASETS FOR THESIS EXPERIMENTS")
    print("="*70)
    print(f"Train size: {TRAIN_SIZE}")
    print(f"Val size: {VAL_SIZE}")
    print(f"Test size: {TEST_SIZE}")
    print(f"GCS bucket: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print("="*70)
    
    # Load processor
    print("\n Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )
    
    # Preprocess and save to GCS
    print("\n Preprocessing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_vqav2_datasets_preprocessed_ultra_lean(
      processor=processor,
      train_size=TRAIN_SIZE,
      val_size=VAL_SIZE,
      test_size=TEST_SIZE,
      gcs_bucket=GCS_BUCKET,
      gcs_prefix=GCS_PREFIX,
      save_frequency=100
  )
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"\nSaved to: gs://{GCS_BUCKET}/{GCS_PREFIX}/")
    print("\nYou can now run validation experiments - they will load instantly!")
    print("="*70)