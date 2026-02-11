from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import torch
from torchvision import transforms as T

def _extract_embeddings(images, device):


def generate_image_clusters(
    images_folder: str,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
    cluster_selection_method: str,
    use_pca: bool,
    pca_components: int,
    separate_outliers: bool,
    device: str
):
    images_folder = Path(images_folder)
    images = sorted(images_folder.glob('*.jpg'))
    num_images = len(images)
    if num_images == 0:
        print(f'No images found in the specified folder {images_folder}.')
        return
    print(f'Found {num_images} images in the specified folder {images_folder}.')

    embeddings_file = images_folder / '_embeddings.npy'
    labels_file = images_folder / '_cluster_labels.csv'
    viz_file = images_folder / '_cluster_visualization.png'

    if embeddings_file.exists() or labels_file.exists() or viz_file.exists():
        print(f'Embeddings and/or labels and/or visualization already exist in {images_folder}. Skipping clustering.')
        return

    transforms = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])




    embeddings = _extract_embeddings(images, device)
    np.save(embeddings_file, embeddings)
    print(f'Embeddings saved to {embeddings_file}.')

    effective_pca = min(pca_components, len(images), embeddings.shape[1]) if use_pca else embeddings.shape[1]

    cluster_labels, was_noise = _cluster_embeddings(
        embeddings=embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        pca_components=effective_pca,
        separate_outliers=separate_outliers,
    )

    filenames = [image.name for image in images]
    df = pd.DataFrame({'filename': filenames, 'cluster': cluster_labels})
    df.to_csv(labels_file, index=False)
    print(f'Cluster labels saved to {labels_file}.')

    _visualize_clusters(images, cluster_labels, viz_file, was_noise=was_noise)

    print(f'\nClustering completed for {images_folder}:')
    print(f' _embeddings.npy ({len(embeddings)} embeddings)')
    print(f' _cluster_labels.csv ({len(df)} rows)')
    print(f' _cluster_visualization.png')



def group_images_into_cluster_folders(images_folder: str, cluster_labels_file: str):
    cluster_labels_df = pd.read_csv(f'{images_folder}{cluster_labels_file}')
    unique_clusters = cluster_labels_df['cluster'].unique()
    for cluster_id in unique_clusters:
        Path(f'{images_folder}cluster_{cluster_id}').mkdir(exist_ok=True)
    moved = 0
    for image_id, cluster_id in cluster_labels_df.itertuples(index=False):
        shutil.move(f'{images_folder}{image_id}.jpg', f'{images_folder}cluster_{cluster_id}/{image_id}.jpg')
        moved += 1
    print(f'Moved {moved} images into {len(unique_clusters)} folders.')

def ungroup_images_in_folder(images_folder: str):
    images_folder = Path(images_folder)
    subfolders = [item for item in images_folder.iterdir() if item.is_dir()]
    moved = 0
    deleted = 0
    for subfolder in subfolders:
        for file in subfolder.iterdir():
            shutil.move(str(file), str(images_folder / file.name))
            moved += 1
        subfolder.rmdir()
        deleted += 1
    print(f"Moved {moved} files back to the main folder. Deleted {deleted} subfolders.")
