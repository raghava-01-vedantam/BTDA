from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
OUTPUT = TEST_DIR / "sample_index.json"


def mask_pixel_count(mask_path: Path) -> int:
    arr = np.array(Image.open(mask_path).convert("L"))
    return int((arr > 127).sum())


def main() -> None:
    img_dir = TEST_DIR / "images"
    mask_dir = TEST_DIR / "masks"

    if not img_dir.exists():
        print(f"Test images not found: {img_dir}")
        sys.exit(1)

    entries = []
    for img_path in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            continue
        count = mask_pixel_count(mask_path)
        entries.append({
            "stem": img_path.stem,
            "image": str(img_path.relative_to(PROJECT_ROOT)),
            "mask": str(mask_path.relative_to(PROJECT_ROOT)),
            "pixels": count,
            "patient": img_path.stem.rsplit("_", 1)[0],
        })

    positive = [e for e in entries if e["pixels"] > 0]
    negative = [e for e in entries if e["pixels"] == 0]

    positive.sort(key=lambda e: e["pixels"])

    n_want = 10
    step = max(1, len(positive) // n_want)
    candidate_indices = list(range(0, len(positive), step))[:n_want]

    seen_patients: set[str] = set()
    selected_positive: list[dict] = []
    for idx in candidate_indices:
        e = positive[idx]
        if e["patient"] not in seen_patients:
            seen_patients.add(e["patient"])
            selected_positive.append(e)
    for e in positive:
        if len(selected_positive) >= n_want:
            break
        if e["patient"] not in seen_patients:
            seen_patients.add(e["patient"])
            selected_positive.append(e)

    selected_positive.sort(key=lambda e: e["pixels"])
    n_sel = len(selected_positive)

    def label_for(i: int, total: int) -> str:
        if total == 1:
            return "Tumor"
        if i < total // 3:
            return "Tumor, small"
        if i < 2 * total // 3:
            return "Tumor, medium"
        return "Tumor, large"

    samples = []
    for i, e in enumerate(selected_positive):
        samples.append({
            "label": label_for(i, n_sel),
            "image": e["image"],
            "mask": e["mask"],
            "pixels": e["pixels"],
            "positive": True,
        })

    for e in negative[:3]:
        samples.append({
            "label": "No tumor",
            "image": e["image"],
            "mask": e["mask"],
            "pixels": 0,
            "positive": False,
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"Wrote {len(samples)} samples to {OUTPUT}")


if __name__ == "__main__":
    main()
