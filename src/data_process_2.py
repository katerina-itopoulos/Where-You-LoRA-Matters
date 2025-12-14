from pathlib import Path
import numpy as np
from datasets import load_dataset

# -----------------------
# CONFIG (edit these)
# -----------------------
SRC = {
    "train": "/mnt/nvme/your_cached/train/train_shard_*.parquet",
    "val":   "/mnt/nvme/your_cached/val/val_shard_*.parquet",
    "test":  "/mnt/nvme/your_cached/test/test_shard_*.parquet",
}

OUT_BASE = Path("/mnt/nvme/qwen_ready")
MAX_LEN = 2048          # pick what you train with (1024/2048/4096)
PAD_ID = 0              # set to your tokenizer pad_token_id if different
SHARD_SIZE = 2048       # rows per output shard
PIXEL_DTYPE = np.float16

def pad_list(seq, max_len, pad_value):
    n = len(seq)
    if n >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - n)

def transform(ex):
    # text
    input_ids = pad_list(ex["input_ids"], MAX_LEN, PAD_ID)
    attn = pad_list(ex["attention_mask"], MAX_LEN, 0)

    # labels
    if "labels" in ex and ex["labels"] is not None:
        labels = pad_list(ex["labels"], MAX_LEN, -100)
    else:
        labels = input_ids[:]  # default LM labels

    out = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }

    # vision (already precomputed)
    if "pixel_values" in ex and ex["pixel_values"] is not None:
        out["pixel_values"] = np.asarray(ex["pixel_values"], dtype=PIXEL_DTYPE).tolist()
        out["image_grid_thw"] = np.asarray(ex["image_grid_thw"], dtype=np.int64).tolist()

    return out

def convert_split(split):
    out_dir = OUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("parquet", data_files=SRC[split], split="train")
    print(f"{split}: loaded {len(ds)} rows")

    keep = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    ds2 = ds.map(
        transform,
        remove_columns=[c for c in ds.column_names if c not in keep],
        desc=f"Converting {split} -> qwen_ready"
    )

    n = len(ds2)
    num_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"{split}: writing {num_shards} shards to {out_dir}")

    for i in range(num_shards):
        a = i * SHARD_SIZE
        b = min(a + SHARD_SIZE, n)
        shard = ds2.select(range(a, b))
        shard.to_parquet(str(out_dir / f"{split}_{i:04d}.parquet"))
        print(f"  wrote {split}_{i:04d}.parquet rows={b-a}")

for split in ["train", "val", "test"]:
    convert_split(split)

print("✅ Done. New dataset at:", OUT_BASE)
