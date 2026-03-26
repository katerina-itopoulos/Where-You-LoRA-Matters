import gc
import json
from collections import Counter
from typing import Callable

import gcsfs
import torch
from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MIN_PIXELS = 196 * 32 * 32
MAX_PIXELS = 196 * 32 * 32

GCS_BUCKET = "where_you_lora_matters_thesis"
GCS_PREFIX = "datasets/preprocessed_vqa_lowres"
SHARD_SIZE = 256

VALID_SIZE = 2000
TEST_SIZE = 5000

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
    trust_remote_code=True,
)

fs = gcsfs.GCSFileSystem()


def _extract_pixel_inputs(inputs: dict) -> dict:
    """Extract and convert pixel_values and image_grid_thw from processor outputs."""
    result = {}

    pv = inputs.get("pixel_values")
    if pv is not None:
        if pv.dim() == 5:
            pv = pv[0]
        result["pixel_values"] = pv.to(torch.float16).numpy()

    igt = inputs.get("image_grid_thw")
    if igt is not None:
        if igt.dim() == 2:
            igt = igt[0]
        result["image_grid_thw"] = igt.numpy()

    return result


def format_and_preprocess_train(example: dict) -> dict:
    """Format a training example into tokenized inputs with a single answer."""
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
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return {
        "input_ids": inputs["input_ids"][0].tolist(),
        "attention_mask": inputs["attention_mask"][0].tolist(),
        **_extract_pixel_inputs(inputs),
    }


def format_and_preprocess_eval(example: dict) -> dict:
    """Format an eval example retaining all annotator answers for VQA accuracy scoring."""
    if "answers" in example and isinstance(example["answers"], list):
        all_answers = [ans["answer"] for ans in example["answers"]]
        answer_counts = Counter(all_answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        answer_counts_str = json.dumps(dict(answer_counts))
    else:
        answer = (
            example["answer"][0]
            if isinstance(example.get("answer"), list) and example["answer"]
            else example.get("answer", example.get("multiple_choice_answer", ""))
        )
        most_common_answer = answer
        answer_counts_str = json.dumps({answer: 10})

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": example["image"]},
            {"type": "text", "text": example["question"]},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return {
        "input_ids": inputs["input_ids"][0].tolist(),
        "attention_mask": inputs["attention_mask"][0].tolist(),
        "answer": most_common_answer,
        "answer_counts": answer_counts_str,
        "labels": most_common_answer,
        **_extract_pixel_inputs(inputs),
    }


def process_split_to_shards(
    stream,
    total_size: int,
    split_name: str,
    formatter: Callable[[dict], dict],
) -> None:
    """Process a streaming dataset split into parquet shards saved to GCS."""
    gcs_split_dir = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{split_name}"
    fs_split_prefix = f"{GCS_BUCKET}/{GCS_PREFIX}/{split_name}"

    print(f"\n{'='*70}\nProcessing {split_name}: {total_size} samples\n{'='*70}")

    existing = sorted(fs.glob(f"{fs_split_prefix}/{split_name}_shard_*.parquet"))
    if existing:
        print(f"Deleting {len(existing)} old shards")
        for path in existing:
            fs.rm(path)

    stream_iter = iter(stream)
    num_done = 0
    shard_idx = 0
    remaining = total_size

    while remaining > 0:
        curr = min(SHARD_SIZE, remaining)
        print(f"\n  Shard {shard_idx}: samples {num_done + 1}-{num_done + curr}")

        chunk = []
        for _ in range(curr):
            try:
                ex = next(stream_iter)
            except StopIteration:
                print("  Stream ended early")
                break
            try:
                chunk.append(formatter(ex))
            except Exception as e:
                print(f"  Skipped example: {e}")

        if not chunk:
            print("  Stopping early (no processed examples)")
            break

        out_path = f"{gcs_split_dir}/{split_name}_shard_{shard_idx:04d}.parquet"
        print(f"  Saving shard to {out_path}")
        Dataset.from_list(chunk).to_parquet(out_path)

        num_done += len(chunk)
        remaining = total_size - num_done
        shard_idx += 1

        del chunk
        gc.collect()

    print(f"\n  {split_name} done: {num_done}/{total_size}")


print("\nVALID split")
val_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
process_split_to_shards(val_stream, VALID_SIZE, "valid", format_and_preprocess_eval)
del val_stream
gc.collect()

print("\nTEST split")
test_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True).skip(VALID_SIZE)
process_split_to_shards(test_stream, TEST_SIZE, "test", format_and_preprocess_eval)
del test_stream
gc.collect()

print("\n" + "=" * 70)
print("All splits processed")
print("=" * 70)