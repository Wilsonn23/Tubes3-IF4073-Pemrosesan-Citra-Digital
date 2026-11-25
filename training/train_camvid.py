import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]
IGNORE_INDEX = 255
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLASS_COLORS = np.array(
    [
        [70, 130, 180],
        [190, 190, 190],
        [255, 215, 0],
        [128, 64, 128],
        [244, 35, 232],
        [107, 142, 35],
        [220, 220, 0],
        [190, 153, 153],
        [0, 0, 142],
        [220, 20, 60],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)

CLASS_GROUPS = {
    "Sky": {"Sky"},
    "Building": {"Archway", "Bridge", "Building", "Tunnel", "Wall"},
    "Pole": {"Column_Pole"},
    "Road": {"LaneMkgsDriv", "LaneMkgsNonDriv", "Road"},
    "Pavement": {"ParkingBlock", "RoadShoulder", "Sidewalk"},
    "Tree": {"Tree", "VegetationMisc"},
    "SignSymbol": {"Misc_Text", "SignSymbol", "TrafficCone", "TrafficLight"},
    "Fence": {"Fence"},
    "Car": {"Car", "MotorcycleScooter", "SUVPickupTruck", "Train", "Truck_Bus"},
    "Pedestrian": {"CartLuggagePram", "Child", "Pedestrian"},
    "Bicyclist": {"Bicyclist"},
}


def load_color_mapping(class_dict_path):
    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    original_to_reduced = {
        original: class_to_id[reduced]
        for reduced, originals in CLASS_GROUPS.items()
        for original in originals
    }
    color_to_id = {}
    found_classes = set()
    with class_dict_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            original_name = row["name"].strip()
            found_classes.add(original_name)
            color = tuple(int(row[channel].strip()) for channel in ("r", "g", "b"))
            color_to_id[color] = original_to_reduced.get(
                original_name, IGNORE_INDEX
            )

    missing = set(original_to_reduced) - found_classes
    if missing:
        raise ValueError(
            f"Classes missing from {class_dict_path}: {', '.join(sorted(missing))}"
        )
    return color_to_id


def make_color_lookup(color_to_id):
    lookup = np.full(1 << 24, IGNORE_INDEX, dtype=np.uint8)
    for (red, green, blue), class_id in color_to_id.items():
        lookup[(red << 16) | (green << 8) | blue] = class_id
    return lookup


def encode_label(mask, color_lookup):
    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    color_codes = (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )
    return color_lookup[color_codes]


class CamVidDataset(Dataset):
    def __init__(self, data_dir, split, size, color_lookup, augment=False):
        self.image_dir = data_dir / split
        self.label_dir = data_dir / f"{split}_labels"
        self.size = size
        self.color_lookup = color_lookup
        self.augment = augment
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in {self.image_dir}")

        missing_labels = [
            path.name
            for path in self.image_paths
            if not (self.label_dir / f"{path.stem}_L.png").is_file()
        ]
        if missing_labels:
            raise FileNotFoundError(
                f"{len(missing_labels)} labels are missing from {self.label_dir}; "
                f"first missing image: {missing_labels[0]}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_dir / f"{image_path.stem}_L.png"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(label_path) as source:
            label = source.convert("RGB")

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        width, height = self.size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        label = label.resize((width, height), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGE_MEAN) / IMAGE_STD
        label_array = encode_label(label, self.color_lookup).astype(np.int64)
        return image_tensor, torch.from_numpy(label_array)


def build_model(pretrained=True):
    model = deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        aux_loss=True,
    )
    model.classifier[4] = nn.Conv2d(256, len(CLASS_NAMES), kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, len(CLASS_NAMES), kernel_size=1)
    return model


def compute_metrics(confusion):
    confusion = confusion.float()
    intersection = torch.diag(confusion)
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    iou = intersection / union.clamp(min=1)
    present = union > 0
    mean_iou = iou[present].mean().item() if present.any() else 0.0
    pixel_accuracy = (
        intersection.sum() / confusion.sum().clamp(min=1)
    ).item()
    return mean_iou, pixel_accuracy, iou.cpu().tolist()


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    aspp_pool_batch_norm = model.classifier[0].convs[-1][2]
    if not isinstance(aspp_pool_batch_norm, nn.BatchNorm2d):
        raise TypeError("Expected BatchNorm2d in the DeepLabV3 ASPP pooling branch.")
    total_loss = 0.0
    confusion = torch.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64, device=device
    )

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        # A singleton batch cannot update BatchNorm after ASPP's global pooling.
        aspp_pool_batch_norm.train(training and images.size(0) > 1)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
                logits = F.interpolate(
                    outputs["out"],
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = criterion(logits, labels)
                if training and "aux" in outputs:
                    aux_logits = F.interpolate(
                        outputs["aux"],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss = loss + 0.4 * criterion(aux_logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        valid = labels != IGNORE_INDEX
        bins = torch.bincount(
            len(CLASS_NAMES) * labels[valid] + predictions[valid],
            minlength=len(CLASS_NAMES) ** 2,
        )
        confusion += bins.reshape(len(CLASS_NAMES), len(CLASS_NAMES))

    mean_iou, pixel_accuracy, class_ious = compute_metrics(confusion)
    return total_loss / len(loader.dataset), mean_iou, pixel_accuracy, class_ious


def save_visualizations(model, dataset, device, output_path, count):
    count = min(count, len(dataset))
    if count <= 0:
        return
    model.eval()
    figure, axes = plt.subplots(count, 4, figsize=(16, 4 * count), squeeze=False)

    with torch.inference_mode():
        for index in range(count):
            image_tensor, label = dataset[index]
            logits = model(image_tensor.unsqueeze(0).to(device))["out"]
            prediction = F.interpolate(
                logits,
                size=label.shape,
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)[0].cpu().numpy()

            image = image_tensor.mul(IMAGE_STD).add(IMAGE_MEAN)
            image = image.permute(1, 2, 0).clamp(0, 1).numpy()
            truth = label.numpy()
            truth_rgb = np.zeros((*truth.shape, 3), dtype=np.uint8)
            valid = truth != IGNORE_INDEX
            truth_rgb[valid] = CLASS_COLORS[truth[valid]]
            prediction_rgb = CLASS_COLORS[prediction]
            overlay = (
                0.5 * (image * 255.0) + 0.5 * prediction_rgb
            ).clip(0, 255).astype(np.uint8)

            for axis, view, title in zip(
                axes[index],
                (image, truth_rgb, prediction_rgb, overlay),
                ("Input", "Ground truth (11 classes)", "Prediction", "Overlay"),
            ):
                axis.imshow(view)
                axis.set_title(title)
                axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved test visualizations to {output_path}")


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained DeepLabV3 on the CamVid dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=project_dir / "CamVid")
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "runs" / "camvid_deeplabv3"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", type=int, default=5)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.height < 1 or args.width < 1:
        raise ValueError("epochs, batch size, height, and width must be positive.")
    if args.num_workers < 0 or args.visualize < 0:
        raise ValueError("num-workers and visualize must not be negative.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = args.data_dir.resolve()
    class_dict_path = data_dir / "class_dict.csv"
    if not class_dict_path.is_file():
        raise FileNotFoundError(f"CamVid class dictionary not found: {class_dict_path}")
    color_to_id = load_color_mapping(class_dict_path)
    color_lookup = make_color_lookup(color_to_id)
    size = (args.width, args.height)
    datasets = {
        split: CamVidDataset(
            data_dir,
            split,
            size,
            color_lookup,
            augment=(split == "train"),
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    print(
        "Split sizes: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items())
    )
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pth"
    model = build_model(pretrained=not args.evaluate_only).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.evaluate_only:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Train the model first."
            )
    else:
        backbone_params = list(model.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_params}
        head_params = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in backbone_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 1e-4},
                {"params": head_params, "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )
        best_val_iou = -1.0
        history = []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_iou, _, _ = run_epoch(
                model, loaders["train"], device, criterion, optimizer, scaler
            )
            val_loss, val_iou, val_accuracy, _ = run_epoch(
                model, loaders["val"], device, criterion
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_miou": train_iou,
                    "val_loss": val_loss,
                    "val_miou": val_iou,
                    "val_pixel_accuracy": val_accuracy,
                }
            )
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss={train_loss:.4f}, mIoU={train_iou:.4f} | "
                f"val loss={val_loss:.4f}, mIoU={val_iou:.4f}, "
                f"pixel accuracy={val_accuracy:.4f}"
            )

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "class_colors": CLASS_COLORS.tolist(),
                        "image_size": [args.height, args.width],
                        "epoch": epoch,
                        "val_miou": val_iou,
                    },
                    checkpoint_path,
                )
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        print(f"Saved best checkpoint to {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_iou, test_accuracy, class_ious = run_epoch(
        model, loaders["test"], device, criterion
    )
    print(
        f"Test loss={test_loss:.4f}, mIoU={test_iou:.4f}, "
        f"pixel accuracy={test_accuracy:.4f}"
    )
    print("Test IoU by class:")
    for name, value in zip(CLASS_NAMES, class_ious):
        print(f"  {name}: {value:.4f}")
    save_visualizations(
        model,
        datasets["test"],
        device,
        args.output_dir / "test_predictions.png",
        args.visualize,
    )


if __name__ == "__main__":
    main()
