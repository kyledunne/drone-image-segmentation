import pandas as pd
import numpy as np
import cv2

def main():
    train_df = pd.read_csv('data/train.csv')
    train_ids = train_df['ImageID'].to_numpy()
    unique_values = []
    for id in train_ids:
        mask_location = f'data/masks/train/{id}.png'
        # TODO

    # TODO
    ids_with_class_values_df.to_csv('data/ids_with_class_values.csv')

if __name__ == "__main__":
    main()