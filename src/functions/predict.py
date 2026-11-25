import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

project_dir = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "training/train_camvid.py").is_file()
    ),
    None,
)
if project_dir is None:
    raise FileNotFoundError(
        "Could not locate train_camvid.py in this project or its parent folders."
    )
training_dir = project_dir / "training"
sys.path.insert(0, str(training_dir))
train_camvid = importlib.import_module("train_camvid")
CLASS_COLORS = train_camvid.CLASS_COLORS
IMAGE_MEAN = train_camvid.IMAGE_MEAN
IMAGE_STD = train_camvid.IMAGE_STD
build_model = train_camvid.build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict 11-class CamVid segmentation for one image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    for path, description in (
        (args.image, "Input image"),
        (args.checkpoint, "Model checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_colors = np.asarray(
        checkpoint.get("class_colors", CLASS_COLORS), dtype=np.uint8
    )
    if class_colors.shape != (len(CLASS_COLORS), 3):
        raise ValueError(
            "Checkpoint class_colors must contain exactly 11 RGB colors."
        )
    image_size = checkpoint.get("image_size", [360, 480])
    if (
        len(image_size) != 2
        or any(not isinstance(value, int) or value < 1 for value in image_size)
    ):
        raise ValueError("Checkpoint image_size must be [height, width].")
    height, width = image_size

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    original_size = image.size
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    image_tensor = (image_tensor - IMAGE_MEAN).div(IMAGE_STD).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)["out"]
        logits = F.interpolate(
            logits,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        class_map = logits.argmax(dim=1)[0].cpu().numpy()

    original_rgb = np.asarray(image, dtype=np.uint8)
    color_mask = class_colors[class_map]
    overlay = (
        0.5 * original_rgb.astype(np.float32)
        + 0.5 * color_mask.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_mask, mode="RGB").save(args.mask_output)
    Image.fromarray(overlay, mode="RGB").save(args.overlay_output)
    print(f"Segmentation mask saved to: {args.mask_output}")
    print(f"Overlay saved to: {args.overlay_output}")


if __name__ == "__main__":
    main()
