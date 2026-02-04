"""
Script to calculate class distribution from mask images.
Reads all PNG mask files from data/masks/train and prints the percentage
of pixels belonging to each class.
"""

import os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image


def calculate_class_distribution(masks_dir: str) -> dict:
    """
    Calculate the class distribution across all mask images in a directory.

    Args:
        masks_dir: Path to directory containing mask PNG files

    Returns:
        Dictionary mapping class values to their pixel counts
    """
    masks_path = Path(masks_dir)
    png_files = list(masks_path.glob("*.png"))

    if not png_files:
        print(f"No PNG files found in {masks_dir}")
        return {}

    print(f"Found {len(png_files)} mask files")

    total_counts = Counter()

    for mask_file in png_files:
        img = Image.open(mask_file)
        mask_array = np.array(img)
        unique, counts = np.unique(mask_array, return_counts=True)
        for val, count in zip(unique, counts):
            total_counts[val] += count

    return total_counts


def print_distribution(counts: dict) -> None:
    """Print the class distribution as percentages."""
    total_pixels = sum(counts.values())

    if total_pixels == 0:
        print("No pixels found")
        return

    print(f"\nTotal pixels: {total_pixels:,}")
    print("\nClass Distribution:")
    print("-" * 40)

    for class_val in sorted(counts.keys()):
        count = counts[class_val]
        percentage = (count / total_pixels) * 100
        print(f"Class {class_val}: {percentage:>6.2f}% ({count:,} pixels)")


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    masks_dir = project_dir / "data" / "masks" / "train"

    print(f"Reading masks from: {masks_dir}")

    if not masks_dir.exists():
        print(f"Error: Directory does not exist: {masks_dir}")
        return

    counts = calculate_class_distribution(masks_dir)

    if counts:
        print_distribution(counts)


if __name__ == "__main__":
    main()
