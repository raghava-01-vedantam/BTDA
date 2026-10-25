from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.losses import DiceLoss, DiceBCELoss
from model.unet import build_unet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net on LGG MRI dataset.")
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch-size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--loss",           type=str,   default="dice",
                   choices=["dice", "dice_bce"])
    p.add_argument("--scheduler",      action="store_true",
                   help="ReduceLROnPlateau on val Dice (factor=0.5, patience=3)")
    p.add_argument("--patience",       type=int,   default=8,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--amp",            action="store_true", default=True,
                   help="Mixed precision (default: enabled on CUDA)")
    p.add_argument("--no-amp",         dest="amp", action="store_false")
    p.add_argument("--max-samples",    type=int,   default=0,
                   help="Subsample training set (0 = full dataset)")
    p.add_argument("--run-name",       type=str,   default="")
    p.add_argument("--data-dir",       type=str,   default="./data/processed")
    p.add_argument("--checkpoint-dir", type=str,   default="./model/checkpoints")
    p.add_argument("--num-workers",    type=int,   default=0)
    return p.parse_args()


# Dataset

class MRISegmentationDataset(data.Dataset):
    """Paired MRI image / mask PNGs from a processed split directory."""

    def __init__(
        self,
        split_dir: str | Path,
        transform: A.Compose | None = None,
        max_samples: int = 0,
    ) -> None:
        self.transform = transform
        split_path = Path(split_dir)
        img_dir  = split_path / "images"
        mask_dir = split_path / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
            else:
                print(f"[WARN] Mask not found for {img_path.name} — skipping.")

        if not self.samples:
            raise RuntimeError(f"No valid image/mask pairs found in {split_dir}")

        if max_samples > 0 and max_samples < len(self.samples):
            np.random.seed(42)
            idxs = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask  = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)

        return image, mask


# Augmentation pipelines

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform() -> A.Compose:
    return A.Compose([
        A.Resize(256, 256),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ElasticTransform(alpha=40, sigma=5, p=0.3),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


# Metrics helpers

def _dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute mean Dice over a batch (after sigmoid + threshold at 0.5)."""
    probs = torch.sigmoid(preds)
    pred_bin = (probs > 0.5).float()
    tp = (pred_bin * targets).sum().item()
    fp = (pred_bin * (1 - targets)).sum().item()
    fn = ((1 - pred_bin) * targets).sum().item()
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    return dice


def _iou_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    probs = torch.sigmoid(preds)
    pred_bin = (probs > 0.5).float()
    tp = (pred_bin * targets).sum().item()
    fp = (pred_bin * (1 - targets)).sum().item()
    fn = ((1 - pred_bin) * targets).sum().item()
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    return iou


# One epoch helpers

def train_one_epoch(
    model: torch.nn.Module,
    loader: data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler,
) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks  = masks.to(device, dtype=torch.float32)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (val_loss, val_dice, val_iou)."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0
    n_batches  = 0

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks  = masks.to(device, dtype=torch.float32)
        logits = model(images)
        loss   = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        total_dice += _dice_score(logits, masks)
        total_iou  += _iou_score(logits, masks)
        n_batches  += 1

    val_loss = total_loss / len(loader.dataset)
    val_dice = total_dice / n_batches if n_batches else 0.0
    val_iou  = total_iou  / n_batches if n_batches else 0.0
    return val_loss, val_dice, val_iou


# Loss curve

def plot_loss_curve(
    train_losses: list[float],
    val_losses:   list[float],
    val_dices:    list[float],
    save_path: str | Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="#F44336", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_dices, label="Val Dice", color="#4CAF50", linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice")
    ax2.set_title("Val Dice Curve"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss curve saved -> {save_path}")


# Checkpoint promotion

def promote_if_better(
    run_checkpoint: Path,
    run_val_dice:   float,
    checkpoint_dir: Path,
    run_name:       str,
) -> None:
    promoted_json = checkpoint_dir / "promoted.json"
    best_path     = checkpoint_dir / "best_model.pth"
    baseline_path = checkpoint_dir / "baseline_epoch9.pth"

    # First-time setup: back up existing checkpoint
    if not promoted_json.exists():
        if best_path.exists() and not baseline_path.exists():
            shutil.copy2(best_path, baseline_path)
            print(f"[INFO] Backed up existing checkpoint to {baseline_path.name}")
        baseline_dice = 0.8450  # known baseline val Dice from previous training
        data_json = {
            "promoted_run":    "baseline",
            "val_dice":        baseline_dice,
            "model_backbone":  "resnet34",
            "model_iou_score": 0.7316,
            "trained_date":    "2026-08-22",
        }
        with open(promoted_json, "w") as f:
            json.dump(data_json, f, indent=2)
        print(f"[INFO] Initialized promoted.json with baseline val_dice={baseline_dice}")

    with open(promoted_json) as f:
        promoted = json.load(f)

    current_best_dice = float(promoted.get("val_dice", 0.0))

    if run_val_dice > current_best_dice:
        shutil.copy2(run_checkpoint, best_path)
        promoted["promoted_run"]    = run_name
        promoted["val_dice"]        = run_val_dice
        promoted["model_backbone"]  = "resnet34"
        promoted["model_iou_score"] = 0.0  # updated by evaluate.py
        promoted["trained_date"]    = date.today().isoformat()
        with open(promoted_json, "w") as f:
            json.dump(promoted, f, indent=2)
        print(f"[PROMOTED] New best val Dice {run_val_dice:.4f} > {current_best_dice:.4f}")
        print(f"           Copied to {best_path}")
    else:
        print(f"[INFO] Run val Dice {run_val_dice:.4f} <= promoted {current_best_dice:.4f}; not promoted.")


# Main

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        print("[INFO] Mixed precision (AMP) enabled")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ---- Run directory ----
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"run_{args.loss}_ep{args.epochs}"
    run_dir  = checkpoint_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_checkpoint = run_dir / "best_model.pth"

    # ---- Datasets ----
    data_dir = Path(args.data_dir)
    train_ds = MRISegmentationDataset(
        data_dir / "train",
        transform=get_train_transform(),
        max_samples=args.max_samples,
    )
    val_ds = MRISegmentationDataset(
        data_dir / "val",
        transform=get_val_transform(),
    )

    def _make_loader(ds, shuffle):
        try:
            loader = data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=False,
            )
            # Test with one batch
            _ = next(iter(loader))
            return loader
        except Exception:
            if args.num_workers > 0:
                print("[WARN] Workers failed, retrying with num_workers=0")
                return data.DataLoader(
                    ds, batch_size=args.batch_size, shuffle=shuffle,
                    num_workers=0, pin_memory=False,
                )
            raise

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader   = _make_loader(val_ds,   shuffle=False)

    print(f"[INFO] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"[INFO] Loss: {args.loss} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    # ---- Model ----
    model = build_unet().to(device)

    # ---- Loss ----
    if args.loss == "dice_bce":
        criterion = DiceBCELoss(smooth=1.0)
    else:
        criterion = DiceLoss(smooth=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3,
        )
        print("[INFO] ReduceLROnPlateau scheduler enabled (patience=3, factor=0.5)")

    # ---- History ----
    history_path = run_dir / "history.csv"
    history_file = open(history_path, "w", newline="", encoding="utf-8")
    history_writer = csv.writer(history_file)
    history_writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr"])

    # ---- Training loop ----
    train_losses: list[float] = []
    val_losses:   list[float] = []
    val_dices:    list[float] = []

    best_val_dice = -1.0
    best_epoch    = 0
    no_improve    = 0

    print(f"\n[INFO] Starting training ({run_name}) for {args.epochs} epoch(s) ...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp, scaler
        )
        val_loss, val_dice, val_iou = validate_one_epoch(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        history_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_dice:.6f}", f"{val_iou:.6f}", f"{current_lr:.2e}",
        ])
        history_file.flush()

        print(
            f"Epoch [{epoch:>3}/{args.epochs}]  "
            f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
            f"Dice: {val_dice:.4f}  IoU: {val_iou:.4f}  LR: {current_lr:.2e}"
        )

        if scheduler is not None:
            scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch    = epoch
            no_improve    = 0
            torch.save(
                {
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":           val_loss,
                    "val_dice":           val_dice,
                    "val_iou":            val_iou,
                    "run_name":           run_name,
                },
                run_checkpoint,
            )
            print(f"           => Best checkpoint (Dice={val_dice:.4f}) saved")
        else:
            no_improve += 1

        # Early stopping
        if args.patience > 0 and no_improve >= args.patience:
            print(f"\n[EARLY STOP] No val Dice improvement for {args.patience} epochs.")
            break

    history_file.close()

    # ---- Save artefacts ----
    plot_loss_curve(train_losses, val_losses, val_dices, run_dir / "loss_curve.png")

    meta = {
        "run_name":      run_name,
        "loss":          args.loss,
        "epochs_run":    len(train_losses),
        "best_epoch":    best_epoch,
        "best_val_dice": best_val_dice,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "amp":           use_amp,
        "scheduler":     args.scheduler,
        "patience":      args.patience,
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Best val Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"       Run dir: {run_dir.resolve()}")

    # ---- Promotion ----
    if run_checkpoint.exists():
        promote_if_better(run_checkpoint, best_val_dice, checkpoint_dir, run_name)
    else:
        print("[WARN] No checkpoint saved (possibly 0 improvements?).")


if __name__ == "__main__":
    main()
