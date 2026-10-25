import argparse
import os
import shutil
import zipfile
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the LGG MRI Segmentation dataset from a local zip file."
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="./lgg-mri-segmentation.zip",
        help="Path to the downloaded lgg-mri-segmentation.zip (default: ./lgg-mri-segmentation.zip)",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="./raw_data/",
        help="Directory to extract the zip contents into (default: ./raw_data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed/",
        help="Root directory for processed train/val/test splits (default: ./data/processed/)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of patients to hold out for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of patients to hold out for testing (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract the dataset zip into extract_dir if not already done."""
    extract_path = Path(extract_dir)

    # Skip extraction if the directory already exists and is non-empty
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"[INFO] '{extract_dir}' already exists and is non-empty — skipping extraction.")
        return

    zip_path_obj = Path(zip_path)
    if not zip_path_obj.exists():
        raise FileNotFoundError(
            f"Dataset zip not found at '{zip_path}'. "
            "Please download lgg-mri-segmentation.zip from "
            "https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation "
            "and supply its path via --zip-path."
        )

    print(f"[INFO] Extracting '{zip_path}' → '{extract_dir}' ...")
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path_obj, "r") as zf:
        zf.extractall(extract_path)
    print("[INFO] Extraction complete.")


# ---------------------------------------------------------------------------
# Discovery — handle nested kaggle_3m/ structure
# ---------------------------------------------------------------------------

def find_patient_dirs(extract_dir: str) -> list[Path]:
    """
    Walk the extracted directory and return a list of patient-level directories.

    The Kaggle LGG dataset typically extracts as:
        raw_data/
        └── kaggle_3m/
            ├── TCGA_CS_4941_19960909/
            │   ├── TCGA_CS_4941_19960909_1.tif
            │   ├── TCGA_CS_4941_19960909_1_mask.tif
            │   └── ...
            ├── TCGA_CS_4942_19970222/
            └── ...

    A directory is classified as a 'patient folder' if it directly contains
    at least one .tif file that does NOT end with '_mask.tif'.
    """
    extract_path = Path(extract_dir)
    patient_dirs: list[Path] = []

    for candidate in sorted(extract_path.rglob("*")):
        if not candidate.is_dir():
            continue
        tif_images = [
            f for f in candidate.iterdir()
            if f.suffix.lower() == ".tif" and not f.stem.endswith("_mask")
        ]
        if tif_images:
            patient_dirs.append(candidate)

    if not patient_dirs:
        raise RuntimeError(
            f"No patient directories found under '{extract_dir}'. "
            "Make sure the zip extracted correctly and contains .tif files."
        )

    print(f"[INFO] Found {len(patient_dirs)} patient directories.")
    return patient_dirs


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def collect_pairs(patient_dirs: list[Path]) -> dict[str, list[tuple[Path, Path]]]:
    """
    For each patient directory, find all (image_tif, mask_tif) pairs.

    Returns a dict: { patient_id -> [(img_path, mask_path), ...] }
    """
    patient_pairs: dict[str, list[tuple[Path, Path]]] = {}

    for pdir in patient_dirs:
        patient_id = pdir.name
        pairs: list[tuple[Path, Path]] = []

        for tif_file in sorted(pdir.glob("*.tif")):
            # Skip mask files — we'll look them up by convention
            if tif_file.stem.endswith("_mask"):
                continue
            mask_file = tif_file.parent / f"{tif_file.stem}_mask{tif_file.suffix}"
            if mask_file.exists():
                pairs.append((tif_file, mask_file))
            else:
                print(f"[WARN] No mask found for {tif_file} — skipping.")

        if pairs:
            patient_pairs[patient_id] = pairs

    total_pairs = sum(len(v) for v in patient_pairs.values())
    print(f"[INFO] Collected {total_pairs} image/mask pairs across {len(patient_pairs)} patients.")
    return patient_pairs


# ---------------------------------------------------------------------------
# Splitting (patient-level to prevent data leakage)
# ---------------------------------------------------------------------------

def split_patients(
    patient_ids: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Perform a patient-level stratified split into train / val / test.

    val_ratio and test_ratio are fractions of the *total* dataset.
    """
    # First split off test set
    train_val_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )
    # Then split val from the remaining train+val pool
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=adjusted_val_ratio,
        random_state=seed,
        shuffle=True,
    )

    print(
        f"[INFO] Split — train: {len(train_ids)} patients, "
        f"val: {len(val_ids)} patients, "
        f"test: {len(test_ids)} patients"
    )
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Conversion + writing
# ---------------------------------------------------------------------------

def tif_to_png(src: Path, dst: Path) -> None:
    """Open a .tif file and save it as .png at the destination path."""
    img = Image.open(src)
    # Ensure mask images remain single-channel (L mode) or RGB for images
    if "mask" in src.stem:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    img.save(dst, format="PNG")


def write_split(
    split_name: str,
    patient_ids: list[str],
    patient_pairs: dict[str, list[tuple[Path, Path]]],
    output_dir: str,
) -> None:
    """Convert and write all image/mask pairs for a given split."""
    split_root = Path(output_dir) / split_name
    img_dir = split_root / "images"
    mask_dir = split_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for patient_id in patient_ids:
        pairs = patient_pairs.get(patient_id, [])
        for img_tif, mask_tif in pairs:
            stem = f"{patient_id}_{img_tif.stem}"
            tif_to_png(img_tif, img_dir / f"{stem}.png")
            tif_to_png(mask_tif, mask_dir / f"{stem}_mask.png")
            written += 1

    print(f"[INFO]   {split_name:>5}: {written} pairs written → {split_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Step 1 — Extract
    extract_zip(args.zip_path, args.extract_dir)

    # Step 2 — Discover patient directories
    patient_dirs = find_patient_dirs(args.extract_dir)

    # Step 3 — Collect (image, mask) pairs per patient
    patient_pairs = collect_pairs(patient_dirs)
    patient_ids = list(patient_pairs.keys())

    # Step 4 — Patient-level split
    train_ids, val_ids, test_ids = split_patients(
        patient_ids,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Step 5 — Convert .tif → .png and write to output dirs
    print(f"[INFO] Writing processed data to '{args.output_dir}' ...")
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        write_split(split_name, ids, patient_pairs, args.output_dir)

    print("\n[DONE] Dataset preparation complete.")
    print(f"       Processed splits: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
