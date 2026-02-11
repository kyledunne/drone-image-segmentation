"""
Create Per-Cluster Image Folders for Visual Inspection

Copies training images into per-cluster subfolders based on HDBSCAN
cluster assignments, enabling easy visual inspection of cluster quality.

Usage:
    python scripts/create_cluster_folders.py

Output:
    data/clustered_images/cluster_0/
    data/clustered_images/cluster_1/
    ...
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Config:
    train_csv_path: Path
    cluster_labels_path: Path
    images_root_folder: Path
    output_folder: Path


local_config = Config(
    train_csv_path=Path('../data/train.csv'),
    cluster_labels_path=Path('../data_gen/hdbscan_train/cluster_labels.npy'),
    images_root_folder=Path('../data/images/'),
    output_folder=Path('../data/clustered_images/'),
)

kaggle_config = Config(
    train_csv_path=Path('/kaggle/input/opencv-pytorch-segmentation-project-round2/train.csv'),
    cluster_labels_path=Path('/kaggle/working/hdbscan_train/cluster_labels.npy'),
    images_root_folder=Path('/kaggle/input/opencv-pytorch-segmentation-project-round2/imgs/imgs/'),
    output_folder=Path('/kaggle/working/clustered_images/'),
)

config: Config = local_config


def main():
    train_df = pd.read_csv(config.train_csv_path)
    image_ids = train_df["ImageID"].to_numpy()

    cluster_labels = np.load(config.cluster_labels_path)
    assert len(image_ids) == len(cluster_labels), (
        f"Mismatch: {len(image_ids)} image IDs vs {len(cluster_labels)} cluster labels"
    )

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    # Create output directories
    config.output_folder.mkdir(parents=True, exist_ok=True)
    for label in unique_labels:
        (config.output_folder / f"cluster_{label}").mkdir(exist_ok=True)

    # Copy images into cluster folders
    total_copied = 0
    for image_id, label in zip(image_ids, cluster_labels):
        src = config.images_root_folder / f"{image_id}.jpg"
        dst = config.output_folder / f"cluster_{label}" / f"{image_id}.jpg"
        shutil.copy2(src, dst)
        total_copied += 1

    # Summary
    print(f"Clusters: {len(unique_labels)}")
    for label, count in zip(unique_labels, counts):
        print(f"  cluster_{label}: {count} images")
    print(f"Total images copied: {total_copied}")


if __name__ == "__main__":
    main()
