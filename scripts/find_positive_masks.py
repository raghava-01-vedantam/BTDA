from pathlib import Path
from PIL import Image
import numpy as np


def scan_split(split_name: str, root_dir: Path) -> list[tuple[str, int, float]]:
    mask_dir = root_dir / split_name / "masks"
    if not mask_dir.exists():
        return []

    positive_slices = []
    for mask_path in mask_dir.glob("*.png"):
        img = Image.open(mask_path).convert("L")
        arr = np.array(img)
        pos_pixels = int(np.sum(arr > 127))
        if pos_pixels > 0:
            area_cm2 = pos_pixels * 0.013225
            img_filename = mask_path.name.replace("_mask.png", ".png")
            positive_slices.append((img_filename, pos_pixels, area_cm2))

    positive_slices.sort(key=lambda x: x[1], reverse=True)
    return positive_slices


def main():
    root_dir = Path("data/processed")
    if not root_dir.exists():
        return

    for split in ["val", "test"]:
        slices = scan_split(split, root_dir)
        print(f"{split.upper()}: {len(slices)} positive slices")
        for rank, (fname, px, area) in enumerate(slices[:5], start=1):
            print(f"  {rank}. {fname}: {px} px ({area:.2f} cm2)")


if __name__ == "__main__":
    main()
