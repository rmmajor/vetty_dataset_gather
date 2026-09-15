"""The dataset adapter: one annotation row becomes a Sample whose bbox is absolute pixels of the source image."""

from dataclasses import dataclass

from cua.eval.screenspot.config import PREPARED_IMAGE_SIZE


@dataclass
class Sample:
    instruction: str
    bbox_xyxy: tuple  # absolute pixels in the original image
    orig_w: int
    orig_h: int
    data_type: str = "unknown"
    data_source: str = "unknown"
    image: object = None  # PIL.Image, or None when no image is needed


def prepare_sample_image(image):
    if image is None or PREPARED_IMAGE_SIZE == "native":
        return image
    width, height = (int(v) for v in PREPARED_IMAGE_SIZE.lower().split("x"))
    return image.resize((width, height))


def normalize_bbox(raw_bbox, bbox_format, normalized, width, height):
    values = [float(v) for v in raw_bbox]
    if bbox_format == "xywh":
        x, y, w, h = values
        x0, y0, x1, y1 = x, y, x + w, y + h
    elif bbox_format == "xyxy":
        x0, y0, x1, y1 = values
    else:
        raise ValueError(f"Unknown bbox_format {bbox_format!r}. Use 'xywh' or 'xyxy'.")
    if normalized:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    return (x0, y0, x1, y1)


def load_screenspot(cfg):
    # lazy: `datasets` and Pillow are the only heavy deps here, and the mock/smoke path needs neither installed.
    from datasets import load_dataset

    dataset = load_dataset(cfg.dataset, split=cfg.split)
    count = 0
    for row in dataset:
        if cfg.limit is not None and count >= cfg.limit:
            break
        image = row.get("image")
        instruction = row.get("instruction") or row.get("task") or ""
        raw_bbox = row.get("bbox") or row.get("bounding_box")
        if image is None or raw_bbox is None or not instruction:
            continue
        width, height = image.size
        bbox = normalize_bbox(raw_bbox, cfg.bbox_format, cfg.bbox_normalized, width, height)
        yield Sample(
            instruction=instruction,
            bbox_xyxy=bbox,
            orig_w=width,
            orig_h=height,
            data_type=str(row.get("data_type", "unknown")),
            data_source=str(row.get("data_source", row.get("source", "unknown"))),
            image=image,
        )
        count += 1
