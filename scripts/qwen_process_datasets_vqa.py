from transformers import AutoProcessor
import torch
import gc
from datasets import load_dataset, Dataset
import gcsfs
from qwen_vl_utils import process_vision_info

def process_split_to_shards(stream, total_size, split_name, formatter):
    split_dir = f"{GCS_PREFIX}/{split_name}"
    gcs_split_dir = f"gs://{GCS_BUCKET}/{split_dir}"
    fs_split_prefix = f"{GCS_BUCKET}/{split_dir}"

    print(f"\n{'='*70}\nProcessing {split_name}: {total_size} samples\n{'='*70}")

    stream_iter = iter(stream)

    existing = sorted(fs.glob(f"{fs_split_prefix}/{split_name}_shard_*.parquet"))
    if existing:
        print(f"  🗑 Deleting {len(existing)} old shards")
        for path in existing:
            fs.rm(path)
        print("  ✓ Old shards removed")

    num_done = 0
    next_shard_idx = 0
    remaining = total_size

    while remaining > 0:
        curr = min(SHARD_SIZE, remaining)
        print(f"\n  Shard {next_shard_idx}: samples {num_done+1}-{num_done+curr}")

        chunk = []
        for i in range(curr):
            try:
                ex = next(stream_iter)
            except StopIteration:
                print("    ⚠️ Stream ended early")
                break

            try:
                processed = formatter(ex)
                chunk.append(processed)
            except Exception as e:
                print(f"    ⚠️ Skipped example: {e}")
                continue

        if not chunk:
            print("    ⚠️ Stopping early (no processed examples)")
            break

        ds = Dataset.from_list(chunk)
        out_path = f"{gcs_split_dir}/{split_name}_shard_{next_shard_idx:04d}.parquet"
        print(f"    Saving shard to {out_path}")
        ds.to_parquet(out_path)

        num_done += len(ds)
        remaining = total_size - num_done
        next_shard_idx += 1

        del ds, chunk
        gc.collect()

    print(f"\n  ✓ {split_name} done: {num_done}/{total_size}")

def format_and_preprocess_train(example):
    if isinstance(example.get("answer"), list):
        answer = example["answer"][0] if example["answer"] else ""
    else:
        answer = example.get("answer", example.get("multiple_choice_answer", ""))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(answer)}],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

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

    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        pv = inputs["pixel_values"]
        if pv.dim() == 5:
            pv = pv[0]
        pv = pv.to(torch.float16)
        result["pixel_values"] = pv.numpy()

    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        igt = inputs["image_grid_thw"]
        if igt.dim() == 2:
            igt = igt[0]
        result["image_grid_thw"] = igt.numpy()

    return result

def format_and_preprocess_eval(example):
    """
    Preprocess eval samples with VQA metric support.
    Keeps all 10 annotator answers for proper VQA accuracy calculation.
    """
    from collections import Counter
    import json  # ← Add this import!
    
    # Get all 10 answers
    if "answers" in example and isinstance(example["answers"], list):
        # VQAv2 format with 10 annotators
        all_answers = [ans["answer"] for ans in example["answers"]]
        
        # Count answer frequencies
        answer_counts = Counter(all_answers)
        
        # Most common answer (for display/simple metric)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Store answer counts as JSON STRING (for parquet compatibility)
        answer_counts_dict = json.dumps(dict(answer_counts))  # ← Convert to JSON string!
    else:
        # Fallback: single answer format
        if isinstance(example.get("answer"), list):
            answer = example["answer"][0] if example["answer"] else ""
        else:
            answer = example.get("answer", example.get("multiple_choice_answer", ""))
        
        most_common_answer = answer
        answer_counts_dict = json.dumps({answer: 10})  # ← Store as JSON string!
    
    # Build conversation (question only, no answer)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["question"]},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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
        "answer": most_common_answer,  # Most common (for simple metric)
        "answer_counts": answer_counts_dict,  # ← Now a JSON string!
        "labels": most_common_answer,  # For compatibility
    }

    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        pv = inputs["pixel_values"]
        if pv.dim() == 5:
            pv = pv[0]
        pv = pv.to(torch.float16)
        result["pixel_values"] = pv.numpy()

    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        igt = inputs["image_grid_thw"]
        if igt.dim() == 2:
            igt = igt[0]
        result["image_grid_thw"] = igt.numpy()

    return result


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

print("\nTEST split")
val_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
process_split_to_shards(val_stream, VALID_SIZE, "valid", format_and_preprocess_eval)
print("🎉 All splits reprocessed!")


print("\nTEST split")
test_stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
test_stream = test_stream.skip(VALID_SIZE)
process_split_to_shards(test_stream, TEST_SIZE, "test", format_and_preprocess_eval)

del val_stream
del test_stream
gc.collect()

print("\n" + "="*70)
print("🎉 All splits reprocessed!")

