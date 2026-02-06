import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def main():
    train_df = pd.read_csv('data/train.csv')
    train_ids = train_df['ImageID'].to_numpy()

    # Phase 1: Build per-image pixel count table
    rows = []
    for id in train_ids:
        mask_location = f'data/masks/train/{id}.png'
        mask = cv2.imread(mask_location, cv2.IMREAD_GRAYSCALE)
        counts = np.bincount(mask.flatten(), minlength=12)
        rows.append([id] + counts[:12].tolist())

    class_cols = [f'class_{i}' for i in range(12)]
    ids_with_class_values_df = pd.DataFrame(rows, columns=['ImageID'] + class_cols)

    # Sanity check: each row should sum to 1280 * 720 = 921600
    pixel_sums = ids_with_class_values_df[class_cols].sum(axis=1)
    assert (pixel_sums == 921600).all(), f"Some rows don't sum to 921600: {pixel_sums[pixel_sums != 921600]}"
    print(f"Sanity check passed: all {len(ids_with_class_values_df)} rows sum to 921,600 pixels")

    # Phase 2: Stratified train/val split on rare-class presence
    rare_classes = [f'class_{i}' for i in range(1, 8)]
    presence = (ids_with_class_values_df[rare_classes] > 0).astype(int)
    composite_key = presence.apply(lambda row: ''.join(row.astype(str)), axis=1)

    # Merge small groups (fewer than 5 images) into "OTHER"
    group_counts = composite_key.value_counts()
    small_groups = group_counts[group_counts < 5].index
    composite_key = composite_key.replace(small_groups, 'OTHER')

    train_idx, val_idx = train_test_split(
        ids_with_class_values_df.index,
        stratify=composite_key,
        test_size=0.20,
        random_state=42,
    )

    train_split_ids = ids_with_class_values_df.loc[train_idx, 'ImageID'].reset_index(drop=True)
    val_split_ids = ids_with_class_values_df.loc[val_idx, 'ImageID'].reset_index(drop=True)

    # Phase 3: Save outputs
    ids_with_class_values_df.to_csv('data/ids_with_class_values.csv', index=False)
    train_split_ids.to_csv('data/train_split_ids.csv', index=False)
    val_split_ids.to_csv('data/val_split_ids.csv', index=False)

    print(f"\nSplit: {len(train_split_ids)} train / {len(val_split_ids)} val")

    # Phase 4: Print verification summary
    train_data = ids_with_class_values_df.loc[train_idx]
    val_data = ids_with_class_values_df.loc[val_idx]

    print(f"\n{'Class':<10} {'Train Images':>14} {'Train %':>10} {'Val Images':>14} {'Val %':>10}")
    print("-" * 60)

    for i in range(12):
        col = f'class_{i}'
        train_img_count = (train_data[col] > 0).sum()
        val_img_count = (val_data[col] > 0).sum()
        train_pct = train_data[col].sum() / train_data[class_cols].sum().sum() * 100
        val_pct = val_data[col].sum() / val_data[class_cols].sum().sum() * 100
        print(f"class_{i:<4} {train_img_count:>14} {train_pct:>9.2f}% {val_img_count:>14} {val_pct:>9.2f}%")

    print(f"\nSaved: data/ids_with_class_values.csv, data/train_split_ids.csv, data/val_split_ids.csv")


if __name__ == "__main__":
    main()
