"""
Cluster-Class Coverage Analysis for Group-Aware Train/Val Split

Analyzes how the 12 segmentation classes are distributed across HDBSCAN
visual-similarity clusters. Determines whether a cluster-level train/val
split can maintain representation of all classes (especially 7 rare ones)
in both splits.

Usage:
    python scripts/clusters_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

CLASS_NAMES = [
    'background', 'person', 'bike', 'car', 'drone',
    'boat', 'animal', 'obstacle', 'construction',
    'vegetation', 'road', 'sky',
]
CLASS_COLS = [f'class_{i}' for i in range(12)]
RARE_CLASS_INDICES = range(1, 8)
TRAIN_RATIO = 0.80


@dataclass
class Config:
    train_csv_path: Path
    cluster_labels_path: Path
    class_values_csv_path: Path


local_config = Config(
    train_csv_path=Path('../data/train.csv'),
    cluster_labels_path=Path('../data_gen/hdbscan_train/cluster_labels.npy'),
    class_values_csv_path=Path('../data/ids_with_class_values.csv'),
)
kaggle_config = Config(
    train_csv_path=Path('/kaggle/input/opencv-pytorch-segmentation-project-round2/train.csv'),
    cluster_labels_path=Path('/kaggle/working/cluster_labels.npy'),
    class_values_csv_path=Path('/kaggle/working/ids_with_class_values.csv'),
)

config: Config = local_config


def load_data(config: Config) -> pd.DataFrame:
    """Load train IDs, attach cluster labels by position, merge with class pixel counts."""
    train_df = pd.read_csv(config.train_csv_path)
    cluster_labels = np.load(config.cluster_labels_path)
    assert len(train_df) == len(cluster_labels), (
        f"Mismatch: {len(train_df)} images vs {len(cluster_labels)} cluster labels"
    )
    train_df['cluster'] = cluster_labels

    class_values_df = pd.read_csv(config.class_values_csv_path)
    df = train_df.merge(class_values_df, on='ImageID', how='inner')
    assert len(df) == len(train_df), (
        f"Merge lost rows: {len(train_df)} -> {len(df)}"
    )
    return df


def print_per_cluster_summary(df: pd.DataFrame) -> None:
    """Section 1: Per-cluster image count, class presence, and pixel distribution."""
    print("=" * 80)
    print("SECTION 1: Per-Cluster Summary")
    print("=" * 80)

    clusters = sorted(df['cluster'].unique())
    for cluster_id in clusters:
        cdf = df[df['cluster'] == cluster_id]
        n_images = len(cdf)
        print(f"\n--- Cluster {cluster_id} ({n_images} images) ---")

        # Per-class presence (how many images contain at least 1 pixel of the class)
        presence = (cdf[CLASS_COLS] > 0).sum()
        # Per-class pixel share within this cluster
        pixel_totals = cdf[CLASS_COLS].sum()
        cluster_total_pixels = pixel_totals.sum()
        pixel_pct = pixel_totals / cluster_total_pixels * 100 if cluster_total_pixels > 0 else pixel_totals * 0

        print(f"  {'Class':<15} {'Images':>8} {'Pixel %':>10}")
        print(f"  {'-'*35}")
        present_classes = []
        missing_classes = []
        for i in range(12):
            col = CLASS_COLS[i]
            tag = " [RARE]" if i in RARE_CLASS_INDICES else ""
            if presence[col] > 0:
                present_classes.append(CLASS_NAMES[i])
                print(f"  {CLASS_NAMES[i]:<15} {presence[col]:>8} {pixel_pct[col]:>9.2f}%{tag}")
            else:
                missing_classes.append(CLASS_NAMES[i])

        if missing_classes:
            print(f"  Missing: {', '.join(missing_classes)}")


def print_class_coverage(df: pd.DataFrame) -> None:
    """Section 2: Per-class coverage across clusters."""
    print("\n" + "=" * 80)
    print("SECTION 2: Class Coverage Across Clusters")
    print("=" * 80)

    clusters = sorted(df['cluster'].unique())
    for i in range(12):
        col = CLASS_COLS[i]
        tag = " [RARE]" if i in RARE_CLASS_INDICES else ""
        total_images = (df[col] > 0).sum()
        total_pixels = df[col].sum()

        # Which clusters contain this class
        containing_clusters = []
        missing_clusters = []
        for cluster_id in clusters:
            cdf = df[df['cluster'] == cluster_id]
            cluster_presence = (cdf[col] > 0).sum()
            if cluster_presence > 0:
                cluster_pixels = cdf[col].sum()
                containing_clusters.append((cluster_id, int(cluster_presence), int(cluster_pixels)))
            else:
                missing_clusters.append(cluster_id)

        print(f"\n--- {CLASS_NAMES[i]} (class_{i}){tag} ---")
        print(f"  Total: {total_images} images, {total_pixels:,} pixels")
        print(f"  Present in {len(containing_clusters)}/{len(clusters)} clusters")

        # Per-cluster breakdown
        print(f"  {'Cluster':>10} {'Images':>8} {'Pixels':>14} {'% of class':>12}")
        print(f"  {'-'*46}")
        for cluster_id, img_count, px_count in sorted(containing_clusters, key=lambda x: -x[2]):
            pct = px_count / total_pixels * 100 if total_pixels > 0 else 0
            print(f"  {cluster_id:>10} {img_count:>8} {px_count:>14,} {pct:>11.1f}%")

        if missing_clusters:
            print(f"  Missing from clusters: {missing_clusters}")


def print_feasibility_assessment(df: pd.DataFrame) -> None:
    """Section 3: Feasibility of cluster-level 80/20 split."""
    print("\n" + "=" * 80)
    print("SECTION 3: Feasibility Assessment for Cluster-Level Split")
    print("=" * 80)

    clusters = sorted(df['cluster'].unique())

    # 3a: Exclusive clusters — clusters that are the sole source of a rare class
    print("\n--- 3a: Exclusive Clusters (sole source of a rare class) ---")
    found_exclusive = False
    for i in RARE_CLASS_INDICES:
        col = CLASS_COLS[i]
        clusters_with_class = []
        for cluster_id in clusters:
            cdf = df[df['cluster'] == cluster_id]
            if (cdf[col] > 0).any():
                clusters_with_class.append(cluster_id)
        if len(clusters_with_class) == 1:
            found_exclusive = True
            n_images = (df[df['cluster'] == clusters_with_class[0]][col] > 0).sum()
            print(f"  {CLASS_NAMES[i]:<15} ONLY in cluster {clusters_with_class[0]} ({n_images} images)")
    if not found_exclusive:
        print("  None found — no rare class is confined to a single cluster.")

    # 3b: Concentration analysis
    print("\n--- 3b: Rare Class Concentration Analysis ---")
    print(f"  {'Class':<15} {'Clusters':>10} {'Top Cluster %':>15} {'Risk':>8}")
    print(f"  {'-'*50}")
    for i in RARE_CLASS_INDICES:
        col = CLASS_COLS[i]
        total_pixels = df[col].sum()
        if total_pixels == 0:
            print(f"  {CLASS_NAMES[i]:<15} {'N/A':>10} {'N/A':>15} {'N/A':>8}")
            continue

        cluster_pixels = []
        for cluster_id in clusters:
            cdf = df[df['cluster'] == cluster_id]
            px = cdf[col].sum()
            if px > 0:
                cluster_pixels.append((cluster_id, px))

        n_clusters = len(cluster_pixels)
        top_pct = max(px for _, px in cluster_pixels) / total_pixels * 100

        if n_clusters <= 2 or top_pct > 80:
            risk = "HIGH"
        elif n_clusters <= 5 or top_pct > 50:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        print(f"  {CLASS_NAMES[i]:<15} {n_clusters:>10} {top_pct:>14.1f}% {risk:>8}")

    # 3c: Simulated greedy 80/20 cluster-level split
    print("\n--- 3c: Simulated 80/20 Cluster-Level Split (greedy bin-packing) ---")
    total_images = len(df)
    target_train = int(total_images * TRAIN_RATIO)

    # Sort clusters by size descending
    cluster_sizes = [(c, len(df[df['cluster'] == c])) for c in clusters]
    cluster_sizes.sort(key=lambda x: -x[1])

    train_clusters = []
    val_clusters = []
    train_count = 0

    for cluster_id, size in cluster_sizes:
        if train_count + size <= target_train:
            train_clusters.append(cluster_id)
            train_count += size
        else:
            # Check if adding to train gets us closer to target
            gap_if_add = abs((train_count + size) - target_train)
            gap_if_skip = abs(train_count - target_train)
            if gap_if_add < gap_if_skip:
                train_clusters.append(cluster_id)
                train_count += size
            else:
                val_clusters.append(cluster_id)

    train_mask = df['cluster'].isin(train_clusters)
    val_mask = df['cluster'].isin(val_clusters)
    train_df = df[train_mask]
    val_df = df[val_mask]

    print(f"\n  Train: {len(train_clusters)} clusters, {len(train_df)} images ({len(train_df)/total_images:.1%})")
    print(f"  Val:   {len(val_clusters)} clusters, {len(val_df)} images ({len(val_df)/total_images:.1%})")
    print(f"  Train clusters: {sorted(train_clusters)}")
    print(f"  Val clusters:   {sorted(val_clusters)}")

    # Per-class coverage in each split
    print(f"\n  {'Class':<15} {'Train Imgs':>12} {'Val Imgs':>12} {'Train Px%':>12} {'Val Px%':>12} {'Status':>10}")
    print(f"  {'-'*70}")

    missing_from_split = []
    for i in range(12):
        col = CLASS_COLS[i]
        tag = " [RARE]" if i in RARE_CLASS_INDICES else ""

        train_img_count = (train_df[col] > 0).sum()
        val_img_count = (val_df[col] > 0).sum()

        total_class_pixels = df[col].sum()
        if total_class_pixels > 0:
            train_px_pct = train_df[col].sum() / total_class_pixels * 100
            val_px_pct = val_df[col].sum() / total_class_pixels * 100
        else:
            train_px_pct = 0
            val_px_pct = 0

        if train_img_count == 0:
            status = "MISSING-T"
            missing_from_split.append((CLASS_NAMES[i], 'train'))
        elif val_img_count == 0:
            status = "MISSING-V"
            missing_from_split.append((CLASS_NAMES[i], 'val'))
        else:
            status = "OK"

        print(f"  {CLASS_NAMES[i]:<15} {train_img_count:>12} {val_img_count:>12} {train_px_pct:>11.1f}% {val_px_pct:>11.1f}% {status:>10}{tag}")

    # Conclusion
    print("\n  --- Conclusion ---")
    if missing_from_split:
        print("  INFEASIBLE: The following classes are missing from a split:")
        for cls_name, split in missing_from_split:
            print(f"    - {cls_name} missing from {split}")
        print("  A pure cluster-level split cannot guarantee all classes in both splits.")
        print("  Consider: image-level stratified split, or hybrid approach (split clusters")
        print("  that contain exclusive rare classes).")
    else:
        print("  FEASIBLE: All 12 classes are represented in both train and val splits.")
        print("  A cluster-level group-aware split is viable for this dataset.")


def main():
    df = load_data(config)
    n_clusters = df['cluster'].nunique()
    total_images = len(df)
    print(f"Dataset: {total_images} images, {n_clusters} clusters, 12 classes")
    print(f"Rare classes (indices {list(RARE_CLASS_INDICES)}): "
          f"{', '.join(CLASS_NAMES[i] for i in RARE_CLASS_INDICES)}")

    print_per_cluster_summary(df)
    print_class_coverage(df)
    print_feasibility_assessment(df)


if __name__ == "__main__":
    main()
