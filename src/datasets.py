from itertools import islice
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers.image_processing_utils import process_vision_info

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
