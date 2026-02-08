import pandas as pd
from pathlib import Path

CLASS_NAMES = ['background', 'person', 'bike', 'car', 'drone', 'boat', 'animal', 'obstacle', 'construction', 'vegetation', 'road', 'sky']
CLASS_FREQUENCIES = [.2301, .0043, .0006, .0034, .0003, .0002, .0007, .0060, .0379, .3539, .3161, .0464]

def main():
    project_dir = Path(__file__).parent.parent
    class_values_df = pd.read_csv(project_dir / 'data' / 'ids_with_class_values.csv')
    train_ids_df = pd.read_csv(project_dir / 'data' / 'train_split_ids.csv')

    train_data = train_ids_df.merge(class_values_df, left_on='id', right_on='ImageID', how='left')

    class_cols = [f'class_{i}' for i in range(12)]
    presence = train_data[class_cols] > 0
    total_images = len(train_data)

    image_counts = presence.sum()
    image_freq = image_counts / total_images
    inv_image_freq = [1.0 / image_freq[c] if image_freq[c] > 0 else total_images for c in class_cols]

    # Summary table
    print(f"Image-level class frequencies ({total_images} training images):")
    print(f"{'Class':<15} {'Count':>7} {'ImgFreq':>10} {'1/ImgFreq':>10}")
    print("-" * 44)
    for i in range(12):
        print(f"{CLASS_NAMES[i]:<15} {image_counts[class_cols[i]]:>7} {image_freq[class_cols[i]]:>10.4f} {inv_image_freq[i]:>10.2f}")

    weights = []
    for _, row in train_data.iterrows():
        present_inv_freqs = [inv_image_freq[i] for i in range(12) if row[class_cols[i]] > 0]
        weight = max(present_inv_freqs) if present_inv_freqs else 1.0
        weights.append(weight)

    train_data['weight'] = weights

    output = train_data[['id', 'weight']]
    output_path = project_dir / 'data' / 'train_sample_weights.csv'
    output.to_csv(output_path, index=False)

    print(f"\nComputed weights for {len(output)} training images")
    print(f"Weight range: {output['weight'].min():.2f} - {output['weight'].max():.2f}")
    print(f"Median weight: {output['weight'].median():.2f}")

    # Show weight tiers by the rarest class driving each image's weight
    print(f"\nWeight tiers (by rarest class in image):")
    print(f"{'Class':<15} {'1/ImgFreq':>10} {'Images':>8}")
    print("-" * 35)
    for i in range(12):
        w = inv_image_freq[i]
        count = (output['weight'] == w).sum()
        if count > 0:
            print(f"{CLASS_NAMES[i]:<15} {w:>10.2f} {count:>8}")


if __name__ == "__main__":
    main()
