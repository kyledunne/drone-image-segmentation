from PIL import Image
import os

if __name__ == "__main__":
    palette_mode_images = 0
    not_palette_mode_images = 0
    for image in os.listdir("../data/masks"):
        with Image.open('../data/masks/' + image) as im:
            if 'P' in im.mode or 'PA' in im.mode:
                print("The PNG is in palette (indexed) mode.")
                palette_mode_images += 1
            else:
                not_palette_mode_images += 1

    print(f'# of masks in palette mode: {palette_mode_images}')
    print(f'# of masks not in palette mode: {not_palette_mode_images}')