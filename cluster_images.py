# %% [code]
"""
Visual Clustering for Group-Aware Train/Val Split

This script extracts embeddings from training images using ResNet50,
clusters visually similar images, and saves cluster labels for use
with GroupShuffleSplit to prevent data leakage between train/val sets.

Usage:
    python cluster_images.py

Output:
    - data/embeddings.npy: 2048-dim embedding vectors for each image
    - data/cluster_labels.npy: Cluster assignment for each image
    - data/cluster_visualization.png: Grid showing sample images per cluster
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class Config:
    train_csv_path: Path
    images_root_folder: Path
    embeddings_path: Path
    cluster_labels_path: Path
    visualization_path: Path
    device: str

local_config = Config(
    train_csv_path=Path('data/train.csv'),
    images_root_folder=Path('data/images/'),
    embeddings_path=Path('data_gen/embeddings.npy'),
    cluster_labels_path=Path('data_gen/cluster_labels.npy'),
    visualization_path=Path('data_gen/cluster_visualization.png'),
    device='cpu',
)
kaggle_config = Config(
    train_csv_path=Path('/kaggle/input/opencv-pytorch-segmentation-project-round2/train.csv'),
    images_root_folder=Path('/kaggle/input/opencv-pytorch-segmentation-project-round2/imgs/imgs/'),
    embeddings_path=Path('/kaggle/working/embeddings.npy'),
    cluster_labels_path=Path('/kaggle/working/cluster_labels.npy'),
    visualization_path=Path('/kaggle/working/cluster_visualization.png'),
    device='cuda',
)

config: Config = local_config

BATCH_SIZE = 32
IMAGE_SIZE = 224  # ResNet50 input size
PCA_COMPONENTS = 128
DISTANCE_THRESHOLD = 15.0  # For automatic cluster count


class ImageDataset(Dataset):
    """Simple dataset for loading images for embedding extraction."""

    def __init__(self, image_ids, images_dir, transform):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.images_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_id


def create_embedding_model(device):
    """Create ResNet50 model with classification head removed."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final classification layer, keep avgpool output (2048-dim)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings(image_ids, images_dir, device):
    """Extract 2048-dim embeddings from all images using ResNet50."""
    print(f"Extracting embeddings for {len(image_ids)} images...")

    # ImageNet normalization
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(image_ids, images_dir, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = create_embedding_model(device)

    embeddings = []

    with torch.no_grad():
        for batch_images, _ in tqdm(loader, desc="Extracting embeddings"):
            batch_images = batch_images.to(device)
            # Output shape: (batch_size, 2048, 1, 1)
            batch_embeddings = model(batch_images)
            # Flatten to (batch_size, 2048)
            batch_embeddings = batch_embeddings.squeeze(-1).squeeze(-1)
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def cluster_embeddings(embeddings):
    """Cluster embeddings using Agglomerative Clustering with automatic cluster detection."""
    print("Clustering embeddings...")

    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Reduce dimensionality with PCA
    pca = PCA(n_components=PCA_COMPONENTS)
    embeddings_reduced = pca.fit_transform(embeddings_scaled)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: {PCA_COMPONENTS} components explain {explained_var:.1%} of variance")

    # Agglomerative clustering with automatic cluster count
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        linkage="ward",
    )
    cluster_labels = clustering.fit_predict(embeddings_reduced)

    n_clusters = len(np.unique(cluster_labels))
    print(f"Found {n_clusters} clusters")

    # Print cluster size distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    sizes = sorted(counts, reverse=True)
    print(f"Cluster sizes (sorted): {sizes[:20]}{'...' if len(sizes) > 20 else ''}")
    print(f"Min cluster size: {min(sizes)}, Max cluster size: {max(sizes)}, Mean: {np.mean(sizes):.1f}")

    return cluster_labels


def visualize_clusters(image_ids, cluster_labels, images_dir, max_clusters=20, samples_per_cluster=5):
    """Create a grid visualization showing sample images from each cluster."""
    print("Creating cluster visualization...")

    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Sort clusters by size (largest first)
    cluster_sizes = [(c, np.sum(cluster_labels == c)) for c in unique_clusters]
    cluster_sizes.sort(key=lambda x: -x[1])

    # Limit to max_clusters for visualization
    clusters_to_show = [c for c, _ in cluster_sizes[:max_clusters]]
    n_rows = min(max_clusters, n_clusters)

    fig, axes = plt.subplots(n_rows, samples_per_cluster, figsize=(samples_per_cluster * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, cluster_id in enumerate(clusters_to_show):
        cluster_mask = cluster_labels == cluster_id
        cluster_image_ids = image_ids[cluster_mask]
        cluster_size = len(cluster_image_ids)

        # Sample images from this cluster
        sample_indices = np.linspace(0, cluster_size - 1, min(samples_per_cluster, cluster_size), dtype=int)

        for col_idx in range(samples_per_cluster):
            ax = axes[row_idx, col_idx]

            if col_idx < len(sample_indices):
                img_id = cluster_image_ids[sample_indices[col_idx]]
                img_path = images_dir / f"{img_id}.jpg"
                img = Image.open(img_path)
                ax.imshow(img)

                if col_idx == 0:
                    ax.set_ylabel(f"C{cluster_id}\n(n={cluster_size})", fontsize=8)

            ax.axis("off")

    plt.suptitle(f"Sample Images from {n_rows} Largest Clusters (of {n_clusters} total)", fontsize=12)
    plt.tight_layout()
    plt.savefig(config.visualization_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {config.visualization_path}")


def main():
    """Main function to orchestrate embedding extraction, clustering, and visualization."""
    print(f"Using device: {config.device}")

    # Load training image IDs
    train_df = pd.read_csv(config.train_csv_path)
    image_ids = train_df["ImageID"].to_numpy()
    print(f"Found {len(image_ids)} training images")

    # Extract or load embeddings
    if config.embeddings_path.exists():
        print(f"Loading cached embeddings from {config.embeddings_path}")
        embeddings = np.load(config.embeddings_path)
    else:
        embeddings = extract_embeddings(image_ids, config.images_root_folder, config.device)
        np.save(config.embeddings_path, embeddings)
        print(f"Embeddings saved to {config.embeddings_path}")

    # Cluster embeddings
    cluster_labels = cluster_embeddings(embeddings)
    np.save(config.cluster_labels_path, cluster_labels)
    print(f"Cluster labels saved to {config.cluster_labels_path}")

    # Visualize clusters
    visualize_clusters(image_ids, cluster_labels, config.images_root_folder)

    print("\nDone! Files created:")
    print(f"  - {config.embeddings_path}")
    print(f"  - {config.cluster_labels_path}")
    print(f"  - {config.visualization_path}")


if __name__ == "__main__":
    main()
