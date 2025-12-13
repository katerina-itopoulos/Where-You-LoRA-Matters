from pathlib import Path
from datasets import Dataset, load_dataset
from datasets import load_dataset, Dataset
from qwen_vl_utils import process_vision_info
from itertools import islice
from datasets import load_dataset, Dataset, concatenate_datasets
from qwen_vl_utils import process_vision_info
from itertools import islice
import gcsfs
import gc

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as fs
from datasets import Dataset
import pyarrow.fs as fs

def load_parquet_shards_from_gcs(bucket, prefix, split, num_samples=None):
    """
    Load parquet shards from GCS into a HuggingFace Dataset.
    
    Args:
        bucket: GCS bucket name
        prefix: Path prefix (e.g., "datasets/preprocessed_vqa_6k")
        split: "train", "val", or "test"
        num_samples: Optional - take only first N samples
    """
    gcs = fs.GcsFileSystem()
    shard_path = f"{bucket}/{prefix}/{split}/"
    
    # List all parquet files
    file_info = gcs.get_file_info(fs.FileSelector(shard_path, recursive=False))
    parquet_files = [f.path for f in file_info if f.path.endswith('.parquet')]
    parquet_files.sort()  # Ensure consistent ordering
    
    print(f"Found {len(parquet_files)} shards for {split}")
    
    # Read all shards into one table
    tables = []
    total_rows = 0
    
    for filepath in parquet_files:
        table = pq.read_table(f"gs://{filepath}")
        
        if num_samples is not None:
            remaining = num_samples - total_rows
            if remaining <= 0:
                break
            if len(table) > remaining:
                table = table.slice(0, remaining)
        
        tables.append(table)
        total_rows += len(table)
        
        if num_samples is not None and total_rows >= num_samples:
            break
    
    # Concatenate all tables
    full_table = pa.concat_tables(tables)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset(full_table)
    
    print(f"✓ Loaded {len(dataset)} samples for {split}")
    return dataset
    

def cleanup_hf_cache():
    """Clean up HuggingFace cache to free disk space"""
    import shutil

    # HuggingFace cache locations
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "datasets",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]

    print("\n🧹 Cleaning up HuggingFace cache...")
    total_freed = 0

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            size_before = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            size_before_mb = size_before / (1024 * 1024)

            # Remove cache directory
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            total_freed += size_before_mb
            print(f"  Cleared {cache_dir.name}: {size_before_mb:.1f} MB")

    print(f"✓ Total space freed: {total_freed:.1f} MB\n")

def prepare_vqav2_datasets(
    processor,
    train_size=100,
    val_size=20,
    test_size=20,
    min_pixels=256*28*28,
    max_pixels=1280*28*28,
    cache_dir=None,  # Set to None to use temp directory
):
    """
    Prepare VQAv2 datasets with vision-language format

    Args:
        processor: Qwen2VL processor (handles images + text)
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        min_pixels: Minimum image resolution
        max_pixels: Maximum image resolution
        cache_dir: Where to cache dataset (None = temp dir, cleaned after)

    Returns:
        train_dataset, val_dataset, test_dataset

    Note: lmms-lab/VQAv2 has splits: validation, testdev, test (no train split!)
          We use validation for training and testdev for val/test
    """
    print(f"\n📊 Loading VQAv2 dataset with streaming to save disk space...")

    # lmms-lab/VQAv2 only has: validation, testdev, test
    # We'll use validation split for training, testdev for val/test
    train_dataset_stream = load_dataset(
        "lmms-lab/VQAv2",
        split="validation",  # Use validation as training data
        streaming=True,
        cache_dir=cache_dir
    )

    testdev_dataset_stream = load_dataset(
        "lmms-lab/VQAv2",
        split="testdev",  # Use testdev for validation
        streaming=True,
        cache_dir=cache_dir
    )

    # Take only what we need from the stream
    print(f"Taking {train_size} train samples, {val_size} val samples, {test_size} test samples...")

    # Convert streaming datasets to lists (only download what we need)
    train_examples = list(islice(train_dataset_stream, train_size))
    val_examples = list(islice(testdev_dataset_stream, val_size))
    test_examples = list(islice(testdev_dataset_stream.skip(val_size), test_size))

    print(f"✓ Downloaded only {len(train_examples) + len(val_examples) + len(test_examples)} samples")

    def format_vqav2_example(example):
        """
        Format VQAv2 example into Qwen2.5-VL message format

        lmms-lab/VQAv2 structure:
        - image: PIL Image
        - question: str
        - answer: str (or answers list - we'll use the first one)
        """
        # Handle answer format (might be list or single string)
        if isinstance(example.get('answer'), list):
            answer = example['answer'][0] if example['answer'] else ""
        else:
            answer = example.get('answer', example.get('multiple_choice_answer', ''))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example['image']},
                    {"type": "text", "text": example['question']}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer)}]
            }
        ]

        # Process with Qwen2VL processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        # Process images and text
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Extract and convert properly
        # pixel_values shape: (batch, num_tiles, channels, height, width) or (num_tiles, C, H, W)
        # We need to keep all tiles together
        result = {
            "input_ids": inputs['input_ids'][0].long(),
            "attention_mask": inputs['attention_mask'][0].long(),
        }

        # Handle vision inputs carefully
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            pv = inputs['pixel_values']
            # Remove batch dimension if present, keep tiles dimension
            if pv.dim() == 5:  # (batch=1, num_tiles, C, H, W)
                result['pixel_values'] = pv[0]  # (num_tiles, C, H, W)
            else:  # Already (num_tiles, C, H, W)
                result['pixel_values'] = pv

        if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
            igt = inputs['image_grid_thw']
            # Should be (batch=1, 3) or (1, 3)
            if igt.dim() == 2:
                result['image_grid_thw'] = igt[0]  # (3,)
            else:
                result['image_grid_thw'] = igt

        return result

    print("Formatting datasets...")

    # Process examples into formatted datasets
    train_formatted = [format_vqav2_example(ex) for ex in train_examples]
    val_formatted = [format_vqav2_example(ex) for ex in val_examples]
    test_formatted = [format_vqav2_example(ex) for ex in test_examples]

    # Convert to HF Dataset format
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    test_dataset = Dataset.from_list(test_formatted)

    print(f"\n📊 Dataset splits:")
    print(f"  Train: {len(train_dataset)} samples (from validation split)")
    print(f"  Val:   {len(val_dataset)} samples (from testdev split)")
    print(f"  Test:  {len(test_dataset)} samples (from testdev split)")

    return train_dataset, val_dataset, test_dataset


def prepare_vqav2_datasets_preprocessed_ultra_lean(
    processor,
    train_size=1000,
    val_size=200,
    test_size=200,
    gcs_bucket="where_you_lora_matters_thesis",
    gcs_prefix="datasets/preprocessed_vqa",
    save_frequency=100  # Save checkpoint every N samples
):
    """
    Ultra RAM-efficient with smart shard-level resume
    If train is at 3000/5000, resumes from 3000
    """
    
    # GCS paths
    gcs_train_path = f"gs://{gcs_bucket}/{gcs_prefix}/train_{train_size}"
    gcs_val_path = f"gs://{gcs_bucket}/{gcs_prefix}/val_{val_size}"
    gcs_test_path = f"gs://{gcs_bucket}/{gcs_prefix}/test_{test_size}"
    
    fs = gcsfs.GCSFileSystem()
    
    def check_existing_progress(gcs_path, expected_size):
        """Check how many samples already exist"""
        try:
            if fs.exists(gcs_path.replace("gs://", "")):
                existing = Dataset.load_from_disk(gcs_path)
                current_size = len(existing)
                print(f"    Found {current_size}/{expected_size} samples")
                return current_size, existing
            else:
                print(f"    No existing data")
                return 0, None
        except Exception as e:
            print(f"    Error checking: {e}")
            return 0, None
    
    def format_and_preprocess(example):
        """Preprocess single example"""
        if isinstance(example.get('answer'), list):
            answer = example['answer'][0] if example['answer'] else ""
        else:
            answer = example.get('answer', example.get('multiple_choice_answer', ''))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example['image']},
                    {"type": "text", "text": example['question']}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer)}]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": inputs['input_ids'][0].tolist(),
            "attention_mask": inputs['attention_mask'][0].tolist(),
        }
        
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            pv = inputs['pixel_values']
            if pv.dim() == 5:
                pv = pv[0]
            result['pixel_values'] = pv.tolist()
        
        if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
            igt = inputs['image_grid_thw']
            if igt.dim() == 2:
                igt = igt[0]
            result['image_grid_thw'] = igt.tolist()
        
        return result
    
    def process_split_with_resume(stream, total_size, split_name, gcs_path):
        """
        Process split with resume capability
        Resumes from where it left off if partially complete
        """
        print(f"\n{'='*70}")
        print(f"Processing {split_name}: {total_size} samples")
        print(f"{'='*70}")
        
        # Check existing progress
        current_size, existing_dataset = check_existing_progress(gcs_path, total_size)
        
        # If already complete, return it
        if current_size >= total_size:
            print(f"  ✓ {split_name} already complete!")
            return existing_dataset
        
        # If partial, skip to where we left off
        if current_size > 0:
            print(f"  🔄 Resuming from sample {current_size + 1}")
            # Skip already processed samples in stream
            for _ in range(current_size):
                next(islice(stream, 1))
        
        # Process remaining samples
        remaining = total_size - current_size
        num_chunks = (remaining + save_frequency - 1) // save_frequency
        
        for chunk_idx in range(num_chunks):
            chunk_start = current_size + (chunk_idx * save_frequency)
            chunk_end = min(chunk_start + save_frequency, total_size)
            chunk_size = chunk_end - chunk_start
            
            print(f"\n  Chunk {chunk_idx + 1}/{num_chunks}: samples {chunk_start + 1}-{chunk_end}")
            
            # Process chunk
            chunk_examples = []
            for i in range(chunk_size):
                try:
                    example = next(islice(stream, 1))
                    processed = format_and_preprocess(example)
                    chunk_examples.append(processed)
                    
                    if (i + 1) % 20 == 0:
                        print(f"    Processed {i + 1}/{chunk_size}...")
                except StopIteration:
                    print(f"    ⚠️  Stream ended at {i}")
                    break
                except Exception as e:
                    print(f"    ⚠️  Skipped: {e}")
                    continue
            
            if not chunk_examples:
                print(f"    ⚠️  No samples in chunk, stopping")
                break
            
            # Create chunk dataset
            chunk_dataset = Dataset.from_list(chunk_examples)
            
            # Append to existing or create new
            if existing_dataset is not None:
                combined = concatenate_datasets([existing_dataset, chunk_dataset])
                combined.save_to_disk(gcs_path)
                print(f"    ✓ Appended {len(chunk_dataset)} samples (total: {len(combined)})")
                
                # Update existing_dataset reference
                del existing_dataset
                existing_dataset = combined
                del combined
            else:
                chunk_dataset.save_to_disk(gcs_path)
                print(f"    ✓ Created with {len(chunk_dataset)} samples")
                existing_dataset = chunk_dataset
            
            # FREE RAM!
            del chunk_examples, chunk_dataset
            gc.collect()
        
        # Reload final version from GCS
        final = Dataset.load_from_disk(gcs_path)
        print(f"  ✓ Final {split_name}: {len(final)} samples")
        
        del existing_dataset
        gc.collect()
        
        return final
    
    # Check overall progress
    print("🔍 Checking preprocessing progress...")
    
    # Process each split (with resume)
    print("\n" + "="*70)
    print("TRAIN SPLIT")
    train_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    train_dataset = process_split_with_resume(train_stream, train_size, "train", gcs_train_path)
    del train_stream
    gc.collect()
    
    print("\n" + "="*70)
    print("VAL SPLIT")
    val_stream = load_dataset("lmms-lab/VQAv2", split="testdev", streaming=True)
    val_dataset = process_split_with_resume(val_stream, val_size, "val", gcs_val_path)
    del val_stream
    gc.collect()
    
    print("\n" + "="*70)
    print("TEST SPLIT")
    test_stream = load_dataset("lmms-lab/VQAv2", split="testdev", streaming=True)
    test_stream = test_stream.skip(val_size)
    test_dataset = process_split_with_resume(test_stream, test_size, "test", gcs_test_path)
    del test_stream
    gc.collect()
    
    print("\n" + "="*70)
    print("✓ ALL DATASETS COMPLETE")
    print("="*70)
    
    return train_dataset, val_dataset, test_dataset