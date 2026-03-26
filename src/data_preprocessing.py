import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence


def make_conv_full(img: Image.Image, q: str) -> list[dict]:
    """Return a full multimodal conversation with image and question."""
    return [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]


def make_conv_text_only(q: str) -> list[dict]:
    """Return a text-only conversation with the image ablated."""
    return [{"role": "user", "content": [{"type": "text", "text": q}]}]


def make_conv_image_only(img: Image.Image) -> list[dict]:
    """Return an image-only conversation with an empty text prompt."""
    return [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": ""}]}]


def build_convs_from_rows(
    rows: list[dict],
    mode: str = "full",
) -> tuple[list[list[dict]], list[Image.Image], list[str]]:
    """
    Build conversations for a batch of dataset rows under a given modality mode.

    Args:
        rows: List of dataset examples with ``"image"`` and ``"question"`` keys.
        mode: One of ``"full"``, ``"text_only"``, or ``"image_only"``.

    Returns:
        Tuple of ``(convs, imgs, qs)``.
    """
    assert mode in {"full", "text_only", "image_only"}

    convs, imgs, qs = [], [], []
    for ex in rows:
        img, q = ex["image"], ex["question"]
        imgs.append(img)
        qs.append(q)

        if mode == "full":
            content = [{"type": "image", "image": img}, {"type": "text", "text": q}]
        elif mode == "text_only":
            content = [{"type": "text", "text": q}]
        else:
            content = [{"type": "image", "image": img}, {"type": "text", "text": ""}]

        convs.append([{"role": "user", "content": content}])

    return convs, imgs, qs


def build_convs_all_modes(
    rows: list[dict],
) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]], list[Image.Image], list[str]]:
    """Return conversations for all three modality modes alongside images and questions."""
    convs_full, imgs, qs = build_convs_from_rows(rows, mode="full")
    convs_text, _, _ = build_convs_from_rows(rows, mode="text_only")
    convs_image, _, _ = build_convs_from_rows(rows, mode="image_only")
    return convs_full, convs_text, convs_image, imgs, qs


class VLDataCollatorPadTorch:
    """Pad a batch of tokenized vision-language examples into tensors."""

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features],
            batch_first=True,
            padding_value=0,
        )
        attention_mask = pad_sequence(
            [torch.as_tensor(f["attention_mask"], dtype=torch.long) for f in features],
            batch_first=True,
            padding_value=0,
        )

        batch: dict[str, torch.Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}

        if "labels" in features[0]:
            labels = pad_sequence(
                [torch.as_tensor(f["labels"], dtype=torch.long) for f in features],
                batch_first=True,
                padding_value=-100,
            )
            batch["labels"] = labels[:, : input_ids.shape[1]]
        else:
            batch["labels"] = input_ids.clone()

        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.cat(
                [torch.as_tensor(f["pixel_values"]) for f in features], dim=0
            )
            batch["image_grid_thw"] = torch.stack(
                [torch.as_tensor(f["image_grid_thw"], dtype=torch.long) for f in features], dim=0
            )

        return batch