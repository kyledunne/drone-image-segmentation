"""
Sub-clustering tool for refining image clusters.

Extracts DINOv2 ViT-L/14 embeddings and runs HDBSCAN on all .jpg images in a folder,
then optionally sorts them into subfolders by cluster label.

Usage:
    python scripts/subcluster.py cluster <folder_path> [options]
    python scripts/subcluster.py sort <folder_path>
    python scripts/subcluster.py undo <folder_path>

Examples:
    python scripts/subcluster.py cluster data/clustered_images/cluster_8
    python scripts/subcluster.py cluster data/clustered_images/cluster_8 --min-cluster-size 20 --device cuda
    python scripts/subcluster.py sort data/clustered_images/cluster_8
    python scripts/subcluster.py undo data/clustered_images/cluster_8
"""

import argparse
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import hdbscan
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

BATCH_SIZE = 32
IMAGE_WIDTH = 924   # Nearest multiple of 14 preserving 1280:720 aspect ratio
IMAGE_HEIGHT = 518  # DINOv2 standard fine-tuning height (must be divisible by 14)


class ImageDataset(Dataset):
    """Dataset for loading images by file path for embedding extraction."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


def create_embedding_model(device):
    """Create DINOv2 ViT-L/14 embedding model."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings(image_paths, device):
    """Extract 1024-dim embeddings from images using DINOv2 ViT-L/14."""
    print(f"Extracting embeddings for {len(image_paths)} images...")

    transform = T.Compose([
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = 4 if device == "cuda" else 0
    pin_memory = num_workers > 0

    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    model = create_embedding_model(device)

    embeddings = []
    with torch.no_grad():
        for batch_images in tqdm(loader, desc="Extracting embeddings"):
            batch_images = batch_images.to(device)
            # DINOv2 forward returns CLS token: (batch_size, 1024)
            batch_embeddings = model(batch_images)
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def cluster_embeddings(embeddings, min_cluster_size, min_samples,
                       cluster_selection_epsilon, pca_components,
                       cluster_selection_method, separate_outliers=False):
    """Cluster embeddings using StandardScaler -> PCA -> HDBSCAN.

    Args:
        separate_outliers: If True, noise points keep label -1 (sorted into
            an 'outliers/' folder). If False, noise is reassigned to nearest cluster.

    Returns:
        cluster_labels: Cluster assignment for each image
        was_noise: Boolean mask of originally-noise points
    """
    print("Clustering embeddings...")

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=pca_components)
    embeddings_reduced = pca.fit_transform(embeddings_scaled)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: {pca_components} components explain {explained_var:.1%} of variance")

    clustering = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric='euclidean',
    )
    cluster_labels = clustering.fit_predict(embeddings_reduced)

    was_noise = cluster_labels == -1
    n_noise = np.sum(was_noise)
    n_clusters = len(np.unique(cluster_labels[~was_noise]))
    print(f"Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(cluster_labels):.1%})")

    if n_noise > 0 and n_clusters > 0 and not separate_outliers:
        print("Assigning noise points to nearest clusters...")
        clustered_mask = ~was_noise
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(embeddings_reduced[clustered_mask])
        _, indices = nn.kneighbors(embeddings_reduced[was_noise])

        clustered_labels = cluster_labels[clustered_mask]
        cluster_labels[was_noise] = clustered_labels[indices.flatten()]
    elif n_noise > 0 and separate_outliers:
        print(f"Keeping {n_noise} outliers separate (label -1)")

    # Print cluster size distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    sizes = sorted(counts, reverse=True)
    print(f"Cluster sizes (sorted): {sizes[:20]}{'...' if len(sizes) > 20 else ''}")
    print(f"Min: {min(sizes)}, Max: {max(sizes)}, Mean: {np.mean(sizes):.1f}")

    return cluster_labels, was_noise


def visualize_clusters(image_paths, cluster_labels, output_path,
                       was_noise=None, max_clusters=20, samples_per_cluster=5):
    """Create a grid visualization showing sample images per sub-cluster."""
    print("Creating cluster visualization...")

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Sort clusters by size (largest first)
    cluster_sizes = [(c, np.sum(cluster_labels == c)) for c in unique_clusters]
    cluster_sizes.sort(key=lambda x: -x[1])

    clusters_to_show = [c for c, _ in cluster_sizes[:max_clusters]]
    n_rows = min(max_clusters, n_clusters)

    fig, axes = plt.subplots(n_rows, samples_per_cluster,
                             figsize=(samples_per_cluster * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    image_paths = np.array(image_paths)

    for row_idx, cluster_id in enumerate(clusters_to_show):
        cluster_mask = cluster_labels == cluster_id
        total_cluster_size = np.sum(cluster_mask)

        if was_noise is not None:
            vis_mask = cluster_mask & ~was_noise
        else:
            vis_mask = cluster_mask

        cluster_paths = image_paths[vis_mask]
        vis_size = len(cluster_paths)

        if vis_size > 0:
            sample_indices = np.linspace(0, vis_size - 1,
                                         min(samples_per_cluster, vis_size), dtype=int)
        else:
            sample_indices = []

        for col_idx in range(samples_per_cluster):
            ax = axes[row_idx, col_idx]

            if col_idx < len(sample_indices):
                img = Image.open(cluster_paths[sample_indices[col_idx]])
                ax.imshow(img)
                if col_idx == 0:
                    ax.set_ylabel(f"SC{cluster_id}\n(n={total_cluster_size})", fontsize=8)

            ax.axis("off")

    plt.suptitle(f"Sample Images from {n_rows} Sub-clusters (of {n_clusters} total)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {output_path}")


# ── Public API ──────────────────────────────────────────────────────────────

def cluster_folder(
    folder_path: str,
    min_cluster_size=15,
    min_samples=3,
    cluster_selection_epsilon=10.0,
    cluster_selection_method="eom",
    pca_components=256,
    separate_outliers=False,
    device="cpu"
):
    """Run HDBSCAN sub-clustering on all .jpg images in folder_path.

    Outputs (inside folder_path):
        _embeddings.npy            — cached DINOv2 embeddings
        _cluster_labels.csv        — filename -> cluster_label mapping
        _cluster_visualization.png — grid of sample images per cluster
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_paths = sorted(folder_path.glob("*.jpg"))
    if not image_paths:
        print(f"No .jpg files found in {folder_path}")
        return

    print(f"Found {len(image_paths)} images in {folder_path}")

    embeddings_path = folder_path / "_embeddings.npy"
    labels_path = folder_path / "_cluster_labels.csv"
    viz_path = folder_path / "_cluster_visualization.png"

    # Extract or load cached embeddings
    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        if len(embeddings) != len(image_paths):
            print(f"Cached embeddings count ({len(embeddings)}) != image count "
                  f"({len(image_paths)}). Re-extracting...")
            embeddings = extract_embeddings(image_paths, device)
            np.save(embeddings_path, embeddings)
            print(f"Embeddings saved to {embeddings_path}")
    else:
        embeddings = extract_embeddings(image_paths, device)
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved to {embeddings_path}")

    # Clamp PCA components to number of images/features
    effective_pca = min(pca_components, len(image_paths), embeddings.shape[1])
    if effective_pca != pca_components:
        print(f"Clamped PCA components from {pca_components} to {effective_pca}")

    # Cluster
    cluster_labels, was_noise = cluster_embeddings(
        embeddings, min_cluster_size, min_samples,
        cluster_selection_epsilon, effective_pca, cluster_selection_method,
        separate_outliers,
    )

    # Save labels CSV
    filenames = [p.name for p in image_paths]
    df = pd.DataFrame({"filename": filenames, "cluster_label": cluster_labels})
    df.to_csv(labels_path, index=False)
    print(f"Cluster labels saved to {labels_path}")

    # Visualize
    visualize_clusters(image_paths, cluster_labels, viz_path, was_noise=was_noise)

    print(f"\nDone! Files in {folder_path}:")
    print(f"  _embeddings.npy            ({len(embeddings)} embeddings)")
    print(f"  _cluster_labels.csv        ({len(df)} rows)")
    print(f"  _cluster_visualization.png")


def sort_into_clusters(folder_path):
    """Move images into subcluster_<label>/ subfolders based on _cluster_labels.csv."""
    folder_path = Path(folder_path)
    labels_path = folder_path / "_cluster_labels.csv"

    if not labels_path.exists():
        print(f"No _cluster_labels.csv found in {folder_path}. Run 'cluster' first.")
        return

    df = pd.read_csv(labels_path)
    moved = 0

    for _, row in df.iterrows():
        src = folder_path / row["filename"]
        if not src.exists():
            continue
        label = row["cluster_label"]
        if label == -1:
            dest_dir = folder_path / "outliers"
        else:
            dest_dir = folder_path / f"subcluster_{label}"
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(src), str(dest_dir / row["filename"]))
        moved += 1

    # Summary
    subdirs = sorted(folder_path.glob("subcluster_*/"))
    outlier_dir = folder_path / "outliers"
    if outlier_dir.is_dir():
        subdirs.append(outlier_dir)
    print(f"Moved {moved} images into {len(subdirs)} subfolders:")
    for sd in subdirs:
        count = len(list(sd.glob("*.jpg")))
        print(f"  {sd.name}/  ({count} images)")


def undo_sort(folder_path):
    """Move all images from subcluster_*/ subfolders back to parent and remove subfolders."""
    folder_path = Path(folder_path)
    subdirs = sorted(folder_path.glob("subcluster_*/"))
    outlier_dir = folder_path / "outliers"
    if outlier_dir.is_dir():
        subdirs.append(outlier_dir)

    if not subdirs:
        print(f"No subcluster_*/ or outliers/ subfolders found in {folder_path}. Nothing to undo.")
        return

    moved = 0
    for sc_dir in subdirs:
        for img in sc_dir.glob("*.jpg"):
            shutil.move(str(img), str(folder_path / img.name))
            moved += 1
        # Remove the now-empty subfolder
        try:
            sc_dir.rmdir()
        except OSError:
            # Subfolder not empty (non-jpg files remain)
            print(f"  Warning: {sc_dir.name}/ not empty after moving .jpg files, kept as-is")

    print(f"Moved {moved} images back to {folder_path}")
    print(f"Removed {len(subdirs)} subfolders")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    # cluster_folder(
    #     folder_path='../data/clustered_images/cluster_8/outliers',
    #     min_cluster_size=3,
    #     min_samples=2,
    #     cluster_selection_epsilon=45.0,
    #     cluster_selection_method='eom',
    #     pca_components=256,
    #     device='cpu',
    #     separate_outliers=True,
    # )
    sort_into_clusters('../data/clustered_images/cluster_8/outliers')
    # parser = argparse.ArgumentParser(
    #     description="Sub-cluster images in a folder using HDBSCAN",
    # )
    # subparsers = parser.add_subparsers(dest="command", required=True)
    #
    # # cluster
    # p_cluster = subparsers.add_parser("cluster", help="Run HDBSCAN clustering")
    # p_cluster.add_argument("folder_path", type=Path)
    # p_cluster.add_argument("--min-cluster-size", type=int, default=15)
    # p_cluster.add_argument("--min-samples", type=int, default=8)
    # p_cluster.add_argument("--epsilon", type=float, default=10.0)
    # p_cluster.add_argument("--pca-components", type=int, default=128)
    # p_cluster.add_argument("--device", type=str, default="cpu",
    #                        choices=["cpu", "cuda"])
    #
    # # sort
    # p_sort = subparsers.add_parser("sort", help="Move images into subcluster folders")
    # p_sort.add_argument("folder_path", type=Path)
    #
    # # undo
    # p_undo = subparsers.add_parser("undo", help="Move images back from subcluster folders")
    # p_undo.add_argument("folder_path", type=Path)
    #
    # args = parser.parse_args()
    #
    # if args.command == "cluster":
    #     cluster_folder(
    #         args.folder_path,
    #         min_cluster_size=args.min_cluster_size,
    #         min_samples=args.min_samples,
    #         cluster_selection_epsilon=args.epsilon,
    #         pca_components=args.pca_components,
    #         device=args.device,
    #     )
    # elif args.command == "sort":
    #     sort_into_clusters(args.folder_path)
    # elif args.command == "undo":
    #     undo_sort(args.folder_path)


if __name__ == "__main__":
    main()
