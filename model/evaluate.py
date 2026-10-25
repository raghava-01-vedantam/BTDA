from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
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

from model.train import MRISegmentationDataset
from model.unet import build_unet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained U-Net on the test split.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./model/checkpoints/best_model.pth",
        help="Path to the model checkpoint (default: ./model/checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/processed",
        help="Path to the processed data root (default: ./data/processed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binarization (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model/checkpoints",
        help="Directory to save evaluation artifacts (default: ./model/checkpoints)",
    )
    return parser.parse_args()


# Metrics

def compute_metrics(
    preds_bin: np.ndarray,
    targets_bin: np.ndarray,
    smooth: float = 1e-6,
) -> dict[str, float]:
    """Compute binary segmentation metrics."""
    tp = float(np.sum(preds_bin * targets_bin))
    fp = float(np.sum(preds_bin * (1 - targets_bin)))
    fn = float(np.sum((1 - preds_bin) * targets_bin))

    intersection = tp
    union = tp + fp + fn

    iou       = (intersection + smooth) / (union + smooth)
    dice      = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


# Evaluation transform

def get_test_transform() -> A.Compose:
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


# Detailed evaluation runner

@torch.no_grad()
def run_detailed_evaluation(
    model: torch.nn.Module,
    test_ds: MRISegmentationDataset,
    device: torch.device,
    output_dir: Path,
    default_threshold: float = 0.5,
) -> dict[str, any]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, 256, 256)
        all_probs.append(probs)
        all_targets.append(masks.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)[:, 0, :, :]      # (N, 256, 256)
    targets_arr = np.concatenate(all_targets, axis=0)[:, 0, :, :]  # (N, 256, 256)
    n_samples = len(test_ds)

    # 1. Per-slice metrics at default threshold
    preds_bin = (probs_arr >= default_threshold).astype(np.float32)
    per_slice_records = []
    pos_indices = []

    for i in range(n_samples):
        img_path = test_ds.samples[i][0]
        gt_mask = targets_arr[i]
        pred_mask = preds_bin[i]

        gt_pos = bool(np.sum(gt_mask) > 0)
        if gt_pos:
            pos_indices.append(i)

        m = compute_metrics(pred_mask.flatten(), gt_mask.flatten())
        per_slice_records.append({
            "index": i,
            "filename": Path(img_path).name,
            "has_tumor": int(gt_pos),
            "gt_pixels": int(np.sum(gt_mask)),
            "pred_pixels": int(np.sum(pred_mask)),
            "iou": m["iou"],
            "dice": m["dice"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    # Save per_slice_metrics.csv
    csv_path = output_dir / "per_slice_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_slice_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_slice_records)
    print(f"[INFO] Saved per-slice metrics -> {csv_path}")

    # 2. Overall dataset metrics
    all_metrics = compute_metrics(preds_bin.flatten(), targets_arr.flatten())

    # 3. Tumor-positive only metrics
    if pos_indices:
        pos_preds = preds_bin[pos_indices].flatten()
        pos_targets = targets_arr[pos_indices].flatten()
        pos_metrics = compute_metrics(pos_preds, pos_targets)
    else:
        pos_metrics = all_metrics

    # 4. Threshold sweep (0.30 to 0.70 step 0.05)
    sweep_records = []
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]
    for t in thresholds:
        t_preds = (probs_arr >= t).astype(np.float32).flatten()
        t_m = compute_metrics(t_preds, targets_arr.flatten())
        sweep_records.append({
            "threshold": t,
            "iou": t_m["iou"],
            "dice": t_m["dice"],
            "precision": t_m["precision"],
            "recall": t_m["recall"],
        })

    sweep_csv = output_dir / "threshold_sweep.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "iou", "dice", "precision", "recall"])
        writer.writeheader()
        writer.writerows(sweep_records)
    print(f"[INFO] Saved threshold sweep -> {sweep_csv}")

    # 5. Qualitative grid PNG (6 best and 6 worst tumor-positive predictions)
    if len(pos_indices) >= 12:
        pos_records = [per_slice_records[idx] for idx in pos_indices]
        pos_records_sorted = sorted(pos_records, key=lambda x: x["dice"], reverse=True)
        best_6 = pos_records_sorted[:6]
        worst_6 = pos_records_sorted[-6:]
        selected = best_6 + worst_6
        titles = [f"Best #{i+1} (Dice {r['dice']:.3f})" for i, r in enumerate(best_6)] + \
                 [f"Worst #{i+1} (Dice {r['dice']:.3f})" for i, r in enumerate(worst_6)]

        fig, axes = plt.subplots(12, 3, figsize=(10, 36))
        for row, (rec, title) in enumerate(zip(selected, titles)):
            idx = rec["index"]
            img_pil = Image.open(test_ds.samples[idx][0]).convert("RGB").resize((256, 256))
            gt_m = targets_arr[idx]
            pr_m = preds_bin[idx]

            axes[row, 0].imshow(img_pil)
            axes[row, 0].set_title(f"{title}\nOriginal: {rec['filename']}", fontsize=8)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(gt_m, cmap="gray")
            axes[row, 1].set_title(f"Ground Truth ({rec['gt_pixels']} px)", fontsize=8)
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pr_m, cmap="gray")
            axes[row, 2].set_title(f"Prediction ({rec['pred_pixels']} px)", fontsize=8)
            axes[row, 2].axis("off")

        plt.tight_layout()
        grid_path = output_dir / "qualitative_grid.png"
        fig.savefig(grid_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved qualitative grid -> {grid_path}")

    # 6. Save eval_results.md
    md_path = output_dir / "eval_results.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Results — Brain Tumor Segmentation\n\n")
        f.write(f"**Generated**: {timestamp}\n")
        f.write(f"**Total test samples**: {n_samples} (Tumor-positive: {len(pos_indices)}, Negative: {n_samples - len(pos_indices)})\n\n")
        f.write("## Overall Test Set Metrics (All Slices)\n\n")
        f.write("| Metric    | Value  |\n|-----------|--------|\n")
        f.write(f"| IoU       | {all_metrics['iou']:.4f} |\n")
        f.write(f"| Dice      | {all_metrics['dice']:.4f} |\n")
        f.write(f"| Precision | {all_metrics['precision']:.4f} |\n")
        f.write(f"| Recall    | {all_metrics['recall']:.4f} |\n\n")
        f.write("## Tumor-Positive Slices Only\n\n")
        f.write("| Metric    | Value  |\n|-----------|--------|\n")
        f.write(f"| IoU       | {pos_metrics['iou']:.4f} |\n")
        f.write(f"| Dice      | {pos_metrics['dice']:.4f} |\n")
        f.write(f"| Precision | {pos_metrics['precision']:.4f} |\n")
        f.write(f"| Recall    | {pos_metrics['recall']:.4f} |\n\n")
        f.write("> Metrics evaluated with probability threshold = 0.50.\n")
    print(f"[INFO] Saved eval_results.md -> {md_path}")

    return {
        "all_metrics": all_metrics,
        "pos_metrics": pos_metrics,
        "n_samples": n_samples,
        "n_pos": len(pos_indices),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluation Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found at '{checkpoint_path}'.")
        sys.exit(1)

    model = build_unet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[INFO] Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint.get('epoch', '?')})")

    test_ds = MRISegmentationDataset(
        Path(args.data_dir) / "test",
        transform=get_test_transform(),
    )
    print(f"[INFO] Loaded {len(test_ds)} test samples.")

    out_dir = Path(args.output_dir)
    res = run_detailed_evaluation(model, test_ds, device, out_dir, args.threshold)

    print("\n" + "=" * 45)
    print("  EVALUATION SUMMARY")
    print("=" * 45)
    print(f"  All Slices (N={res['n_samples']}):")
    for k, v in res["all_metrics"].items():
        print(f"    {k.capitalize():<12} {v:.4f}")
    print(f"  Tumor-Positive Only (N={res['n_pos']}):")
    for k, v in res["pos_metrics"].items():
        print(f"    {k.capitalize():<12} {v:.4f}")
    print("=" * 45)


if __name__ == "__main__":
    main()
