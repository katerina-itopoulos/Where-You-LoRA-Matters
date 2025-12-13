from typing import Dict, List, Tuple

import torch
from PIL import Image


def make_conv_full(img: Image.Image, q: str) -> List[Dict]:
    return [{"role":"user","content":[{"type":"image","image":img},
                                      {"type":"text","text":q}]}]

def make_conv_text_only(q: str) -> List[Dict]:
    return [{"role":"user","content":[{"type":"text","text":q}]}]

def make_conv_image_only(img: Image.Image) -> List[Dict]:
    return [{"role":"user","content":[{"type":"image","image":img},
                                      {"type":"text","text":""}]}]

def build_convs_from_rows(
    rows: List[Dict],
    mode: str = "full",
) -> Tuple[List[List[Dict]], List[Image.Image], List[str]]:
    assert mode in {"full", "text_only", "image_only"}
    convs, imgs, qs = [], [], []
    for ex in rows:
        img, q = ex["image"], ex["question"]
        imgs.append(img); qs.append(q)

        if mode == "full":
            content = [{"type":"image","image":img}, {"type":"text","text":q}]
        elif mode == "text_only":
            content = [{"type":"text","text":q}]
        else:
            content = [{"type":"image","image":img}, {"type":"text","text":""}]

        convs.append([{"role":"user", "content": content}])

    return convs, imgs, qs

def build_convs_all_modes(rows):
    convs_full,  imgs, qs = build_convs_from_rows(rows, mode="full")
    convs_text,  _,   _   = build_convs_from_rows(rows, mode="text_only")
    convs_image, _,   _   = build_convs_from_rows(rows, mode="image_only")
    return convs_full, convs_text, convs_image, imgs, qs

    
class VLDataCollator:
    """Custom data collator for vision-language models"""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        """Collate batch of vision-language examples"""
        # Extract text fields
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        
        # Extract labels if present (for validation with labels)
        has_labels = 'labels' in features[0]
        if has_labels:
            labels = [f['labels'] for f in features]

        # Pad text sequences
        batch = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )

        # Ensure correct dtypes for text
        batch['input_ids'] = batch['input_ids'].long()
        batch['attention_mask'] = batch['attention_mask'].long()

        # Handle vision inputs - concatenate all tiles from all images
        pixel_values_list = []
        image_grid_thw_list = []

        for f in features:
            if 'pixel_values' in f and f['pixel_values'] is not None:
                pv = f['pixel_values']
                igt = f['image_grid_thw']

                # Ensure they're tensors
                if not torch.is_tensor(pv):
                    pv = torch.tensor(pv)
                if not torch.is_tensor(igt):
                    igt = torch.tensor(igt)

                pixel_values_list.append(pv)  # Each is (num_tiles, C, H, W)
                image_grid_thw_list.append(igt)  # Each is (3,)

        # Concatenate vision inputs if present
        if pixel_values_list:
            # Concatenate all tiles from all images: (total_tiles, C, H, W)
            batch['pixel_values'] = torch.cat(pixel_values_list, dim=0)
            # Stack grid info: (batch_size, 3)
            batch['image_grid_thw'] = torch.stack(image_grid_thw_list, dim=0)

        # Handle labels
        if has_labels:
            # Pad labels with -100 (ignore index)
            max_length = batch['input_ids'].shape[1]
            padded_labels = []
            
            for label_seq in labels:
                # Pad to max_length with -100
                padding_length = max_length - len(label_seq)
                if padding_length > 0:
                    padded_label = label_seq + [-100] * padding_length
                else:
                    padded_label = label_seq[:max_length]
                padded_labels.append(padded_label)
            
            batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        else:
            # No labels in dataset, create from input_ids (for training)
            batch['labels'] = batch['input_ids'].clone().long()

        return batch