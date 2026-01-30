from __future__ import annotations

import pandas as pd
import shutil
from pathlib import Path


def _read_ids(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path)['ImageID'].tolist()

def move_images(
    data_dir: str | Path = "data",
    image_ext: str = ".jpg",
    mask_ext: str = ".png",
) -> dict[str, int]:
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    train_ids = _read_ids(data_dir / "train.csv")
    test_ids = _read_ids(data_dir / "test.csv")

    train_dir = images_dir / "train"
    test_dir = images_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    mask_train_dir = masks_dir / "train"
    mask_test_dir = masks_dir / "test"
    mask_train_dir.mkdir(parents=True, exist_ok=True)
    mask_test_dir.mkdir(parents=True, exist_ok=True)

    moved_train = moved_test = moved_mask_train = moved_mask_test = 0
    missing = skipped = 0

    def move_for_ids(ids: list[str], src_dir: Path, dest_dir: Path, ext: str) -> int:
        nonlocal missing, skipped
        moved = 0
        for image_id in ids:
            src = src_dir / f"{image_id}{ext}"
            dest = dest_dir / src.name
            if dest.exists():
                skipped += 1
                continue
            if not src.exists():
                missing += 1
                continue
            shutil.move(str(src), str(dest))
            moved += 1
        return moved

    moved_train = move_for_ids(train_ids, images_dir, train_dir, image_ext)
    moved_test = move_for_ids(test_ids, images_dir, test_dir, image_ext)
    moved_mask_train = move_for_ids(train_ids, masks_dir, mask_train_dir, mask_ext)
    moved_mask_test = move_for_ids(test_ids, masks_dir, mask_test_dir, mask_ext)

    return {
        "train_moved": moved_train,
        "test_moved": moved_test,
        "mask_train_moved": moved_mask_train,
        "mask_test_moved": moved_mask_test,
        "missing": missing,
        "skipped": skipped,
    }


if __name__ == "__main__":
    stats = move_images()
    print(stats)
