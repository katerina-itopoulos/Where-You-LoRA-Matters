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
    
from torch.nn.utils.rnn import pad_sequence

class VLDataCollatorPadTorch:
    def __call__(self, features):
        input_ids = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.as_tensor(f["attention_mask"], dtype=torch.long) for f in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # labels: pad with -100 if provided, else copy input_ids
        if "labels" in features[0]:
            labels = [torch.as_tensor(f["labels"], dtype=torch.long) for f in features]
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            labels = labels[:, : input_ids.shape[1]]
            batch["labels"] = labels
        else:
            batch["labels"] = input_ids.clone()

        # vision
        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            pv_list = [torch.as_tensor(f["pixel_values"]) for f in features]
            batch["pixel_values"] = torch.cat(pv_list, dim=0)
            batch["image_grid_thw"] = torch.stack(
                [torch.as_tensor(f["image_grid_thw"], dtype=torch.long) for f in features], dim=0
            )

        return batch