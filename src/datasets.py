import gc
import shutil
from itertools import islice
from pathlib import Path

import gcsfs
import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets, load_dataset
from qwen_vl_utils import process_vision_info


def load_parquet_shards_from_gcs(
    bucket: str,
    prefix: str,
    split: str,
    num_samples: int | None = None,
) -> Dataset:
    """Load parquet shards from a GCS prefix into a HuggingFace Dataset."""
    gcs = fs.GcsFileSystem()
    shard_path = f"{bucket}/{prefix}/{split}/"

    file_info = gcs.get_file_info(fs.FileSelector(shard_path, recursive=False))
    parquet_files = sorted(f.path for f in file_info if f.path.endswith(".parquet"))

    print(f"Found {len(parquet_files)} shards for {split}")

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

    dataset = Dataset(pa.concat_tables(tables))
    print(f"Loaded {len(dataset)} samples for {split}")
    return dataset


def cleanup_hf_cache() -> None:
    """Delete HuggingFace dataset and hub caches and recreate empty directories."""
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "datasets",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]

    total_freed = 0.0
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            size_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            total_freed += size_mb
            print(f"Cleared {cache_dir.name}: {size_mb:.1f} MB")

    print(f"Total space freed: {total_freed:.1f} MB")


def prepare_vqav2_datasets(
    processor,
    train_size: int = 100,
    val_size: int = 20,
    test_size: int = 20,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
    cache_dir: str | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Download and format VQAv2 samples into Qwen2.5-VL message format.

    Uses the ``validation`` split for training and ``testdev`` for val/test,
    as ``lmms-lab/VQAv2`` has no dedicated train split.

    Returns:
        Tuple of ``(train_dataset, val_dataset, test_dataset)``.
    """
    print("Loading VQAv2 dataset with streaming...")

    train_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True, cache_dir=cache_dir)
    testdev_stream = load_dataset("lmms-lab/VQAv2", split="testdev", streaming=True, cache_dir=cache_dir)

    train_examples = list(islice(train_stream, train_size))
    val_examples = list(islice(testdev_stream, val_size))
    test_examples = list(islice(testdev_stream.skip(val_size), test_size))

    print(f"Downloaded {len(train_examples) + len(val_examples) + len(test_examples)} samples total")

    def _format_example(example: dict) -> dict:
        answer = (
            example["answer"][0]
            if isinstance(example.get("answer"), list) and example["answer"]
            else example.get("answer", example.get("multiple_choice_answer", ""))
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["question"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
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
            "input_ids": inputs["input_ids"][0].long(),
            "attention_mask": inputs["attention_mask"][0].long(),
        }

        pv = inputs.get("pixel_values")
        if pv is not None:
            result["pixel_values"] = pv[0] if pv.dim() == 5 else pv

        igt = inputs.get("image_grid_thw")
        if igt is not None:
            result["image_grid_thw"] = igt[0] if igt.dim() == 2 else igt

        return result

    print("Formatting datasets...")

    train_dataset = Dataset.from_list([_format_example(ex) for ex in train_examples])
    val_dataset = Dataset.from_list([_format_example(ex) for ex in val_examples])
    test_dataset = Dataset.from_list([_format_example(ex) for ex in test_examples])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def prepare_vqav2_datasets_preprocessed_ultra_lean(
    processor,
    train_size: int = 1000,
    val_size: int = 200,
    test_size: int = 200,
    gcs_bucket: str = "where_you_lora_matters_thesis",
    gcs_prefix: str = "datasets/preprocessed_vqa",
    save_frequency: int = 100,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Preprocess VQAv2 with chunked GCS checkpointing and resume support.

    Resumes from the last saved checkpoint if a split is partially complete.

    Returns:
        Tuple of ``(train_dataset, val_dataset, test_dataset)``.
    """
    gcs_train_path = f"gs://{gcs_bucket}/{gcs_prefix}/train_{train_size}"
    gcs_val_path = f"gs://{gcs_bucket}/{gcs_prefix}/val_{val_size}"
    gcs_test_path = f"gs://{gcs_bucket}/{gcs_prefix}/test_{test_size}"

    gcsfs_client = gcsfs.GCSFileSystem()

    def _check_existing_progress(gcs_path: str, expected_size: int) -> tuple[int, Dataset | None]:
        try:
            if gcsfs_client.exists(gcs_path.replace("gs://", "")):
                existing = Dataset.load_from_disk(gcs_path)
                print(f"Found {len(existing)}/{expected_size} samples")
                return len(existing), existing
            print("No existing data")
            return 0, None
        except Exception as e:
            print(f"Error checking progress: {e}")
            return 0, None

    def _preprocess_example(example: dict) -> dict:
        answer = (
            example["answer"][0]
            if isinstance(example.get("answer"), list) and example["answer"]
            else example.get("answer", example.get("multiple_choice_answer", ""))
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["question"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
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
            "input_ids": inputs["input_ids"][0].tolist(),
            "attention_mask": inputs["attention_mask"][0].tolist(),
        }

        pv = inputs.get("pixel_values")
        if pv is not None:
            result["pixel_values"] = (pv[0] if pv.dim() == 5 else pv).tolist()

        igt = inputs.get("image_grid_thw")
        if igt is not None:
            result["image_grid_thw"] = (igt[0] if igt.dim() == 2 else igt).tolist()

        return result

    def _process_split_with_resume(
        stream,
        total_size: int,
        split_name: str,
        gcs_path: str,
    ) -> Dataset:
        print(f"\n{'='*70}")
        print(f"Processing {split_name}: {total_size} samples")
        print(f"{'='*70}")

        current_size, existing_dataset = _check_existing_progress(gcs_path, total_size)

        if current_size >= total_size:
            print(f"{split_name} already complete")
            return existing_dataset

        if current_size > 0:
            print(f"Resuming from sample {current_size + 1}")
            for _ in range(current_size):
                next(islice(stream, 1))

        remaining = total_size - current_size
        num_chunks = (remaining + save_frequency - 1) // save_frequency

        for chunk_idx in range(num_chunks):
            chunk_start = current_size + chunk_idx * save_frequency
            chunk_end = min(chunk_start + save_frequency, total_size)
            chunk_size = chunk_end - chunk_start

            print(f"Chunk {chunk_idx + 1}/{num_chunks}: samples {chunk_start + 1}-{chunk_end}")

            chunk_examples = []
            for i in range(chunk_size):
                try:
                    example = next(islice(stream, 1))
                    chunk_examples.append(_preprocess_example(example))
                    if (i + 1) % 20 == 0:
                        print(f"  Processed {i + 1}/{chunk_size}...")
                except StopIteration:
                    print(f"  Stream ended at {i}")
                    break
                except Exception as e:
                    print(f"  Skipped sample: {e}")

            if not chunk_examples:
                print("  No samples in chunk, stopping")
                break

            chunk_dataset = Dataset.from_list(chunk_examples)

            if existing_dataset is not None:
                existing_dataset = concatenate_datasets([existing_dataset, chunk_dataset])
                existing_dataset.save_to_disk(gcs_path)
                print(f"  Appended {len(chunk_dataset)} samples (total: {len(existing_dataset)})")
            else:
                chunk_dataset.save_to_disk(gcs_path)
                print(f"  Created with {len(chunk_dataset)} samples")
                existing_dataset = chunk_dataset

            del chunk_examples, chunk_dataset
            gc.collect()

        final = Dataset.load_from_disk(gcs_path)
        print(f"Final {split_name}: {len(final)} samples")
        del existing_dataset
        gc.collect()
        return final

    print("Checking preprocessing progress...")

    train_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    train_dataset = _process_split_with_resume(train_stream, train_size, "train", gcs_train_path)
    del train_stream
    gc.collect()

    val_stream = load_dataset("lmms-lab/VQAv2", split="testdev", streaming=True)
    val_dataset = _process_split_with_resume(val_stream, val_size, "val", gcs_val_path)
    del val_stream
    gc.collect()

    test_stream = load_dataset("lmms-lab/VQAv2", split="testdev", streaming=True).skip(val_size)
    test_dataset = _process_split_with_resume(test_stream, test_size, "test", gcs_test_path)
    del test_stream
    gc.collect()

    print("All datasets complete")
    return train_dataset, val_dataset, test_dataset


def load_dataset_from_config(config, dataset_name: str) -> Dataset | None:
    """Load a dataset from GCS or HuggingFace according to a config object."""
    if dataset_name not in config.DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return None

    dataset_config = config.DATASETS[dataset_name]
    source = dataset_config["source"]

    if source == "gcs":
        return load_preprocessed_dataset(config, dataset_config["path"])
    if source == "huggingface":
        return load_huggingface_dataset(dataset_config)

    print(f"Unknown source: {source}")
    return None


def load_preprocessed_dataset(config, dataset_path: str) -> Dataset | None:
    """Load a preprocessed dataset from GCS parquet shards."""
    gcsfs_client = gcsfs.GCSFileSystem(token="google_default")
    shard_dir = f"{config.GCS_BUCKET}/{config.PREPROCESSED_DIR}/{dataset_path}"

    try:
        shards = [f"gs://{s}" for s in gcsfs_client.ls(shard_dir) if s.endswith(".parquet")]
        if not shards:
            print(f"No shards found for {dataset_path}")
            return None
        print(f"Loading {len(shards)} shards from {dataset_path}...")
        dataset = load_dataset("parquet", data_files=shards, split="train")
        print(f"Loaded {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"Failed to load {dataset_path}: {e}")
        return None


def load_huggingface_dataset(dataset_config: dict) -> Dataset | None:
    """Load a dataset directly from HuggingFace Hub."""
    try:
        dataset = load_dataset(dataset_config["path"], split=dataset_config["split"])
        print(f"Loaded {len(dataset)} samples from HuggingFace")
        return dataset
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        return None