"""
Script to find duplicate images in the training dataset.
Duplicates are identified by comparing pixel values using image hashing.
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


def compute_image_hash(image_path: str) -> str:
    """Compute a hash of the image pixel values."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Use MD5 hash of the raw pixel bytes
    return hashlib.md5(img.tobytes()).hexdigest()


def find_duplicate_images(image_dir: str) -> dict:
    """
    Find all duplicate images in a directory.

    Args:
        image_dir: Path to directory containing images

    Returns:
        Dictionary mapping hash to list of duplicate file paths
    """
    image_dir = Path(image_dir)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {image_dir}")

    # Compute hash for each image
    hash_to_files = defaultdict(list)

    for image_path in tqdm(image_files, desc="Computing image hashes"):
        img_hash = compute_image_hash(str(image_path))
        if img_hash is not None:
            hash_to_files[img_hash].append(image_path.name)

    # Filter to only keep duplicates (more than one file with same hash)
    duplicates = {
        h: files for h, files in hash_to_files.items()
        if len(files) > 1
    }

    return duplicates


def compute_directory_hashes(image_dir: str) -> dict:
    """
    Compute hashes for all images in a directory.

    Args:
        image_dir: Path to directory containing images

    Returns:
        Dictionary mapping hash to list of filenames
    """
    image_dir = Path(image_dir)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {image_dir}")

    hash_to_files = defaultdict(list)

    for image_path in tqdm(image_files, desc=f"Hashing {image_dir.name}"):
        img_hash = compute_image_hash(str(image_path))
        if img_hash is not None:
            hash_to_files[img_hash].append(image_path.name)

    return hash_to_files


def find_train_test_duplicates(train_dir: str, test_dir: str) -> list:
    """
    Find images that exist in both train and test directories with identical pixels.

    Args:
        train_dir: Path to training images directory
        test_dir: Path to test images directory

    Returns:
        List of tuples (train_filename, test_filename) for matching images
    """
    print("Computing hashes for training images...")
    train_hashes = compute_directory_hashes(train_dir)

    print("\nComputing hashes for test images...")
    test_hashes = compute_directory_hashes(test_dir)

    # Find common hashes
    common_hashes = set(train_hashes.keys()) & set(test_hashes.keys())

    matches = []
    for h in common_hashes:
        for train_file in train_hashes[h]:
            for test_file in test_hashes[h]:
                matches.append((train_file, test_file))

    return matches


def main():
    train_dir = "data/images/train"
    test_dir = "data/images/test"

    # # Check for duplicates within training set
    # if os.path.exists(train_dir):
    #     print("=" * 60)
    #     print("CHECKING FOR DUPLICATES WITHIN TRAINING SET")
    #     print("=" * 60)
    #     duplicates = find_duplicate_images(train_dir)
    #
    #     if not duplicates:
    #         print("\nNo duplicate images found within training set.")
    #     else:
    #         print(f"\nFound {len(duplicates)} sets of duplicate images:")
    #         print("-" * 50)
    #
    #         total_duplicates = 0
    #         for i, (hash_val, files) in enumerate(duplicates.items(), 1):
    #             print(f"\nDuplicate set {i} ({len(files)} images):")
    #             for f in files:
    #                 print(f"  - {f}")
    #             total_duplicates += len(files) - 1
    #
    #         print("-" * 50)
    #         print(f"\nSummary:")
    #         print(f"  Total duplicate sets: {len(duplicates)}")
    #         print(f"  Total redundant images: {total_duplicates}")
    # else:
    #     print(f"Error: Directory '{train_dir}' does not exist")

    # Check for duplicates between train and test sets
    print("\n")
    print("=" * 60)
    print("CHECKING FOR DUPLICATES BETWEEN TRAIN AND TEST SETS")
    print("=" * 60)

    if not os.path.exists(train_dir):
        print(f"Error: Directory '{train_dir}' does not exist")
    elif not os.path.exists(test_dir):
        print(f"Error: Directory '{test_dir}' does not exist")
    else:
        matches = find_train_test_duplicates(train_dir, test_dir)

        if not matches:
            print("\nNo identical images found between train and test sets.")
        else:
            print(f"\nFound {len(matches)} identical image pairs:")
            print("-" * 50)
            for train_file, test_file in matches:
                print(f"  Train: {train_file}  <-->  Test: {test_file}")
            print("-" * 50)
            print(f"\nTotal matches: {len(matches)}")


if __name__ == "__main__":
    main()
