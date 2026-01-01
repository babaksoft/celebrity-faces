import os
import shutil

from .config import  config


def copy_images(src_dir, dest_dir, files):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    for file in files:
        shutil.copy(src_dir / file, dest_dir / file)


def ingest():
    for root, _, files in os.walk(config.IMAGE_ROOT):
        dir_name = os.path.split(root)[-1]
        count = len(files)
        if count and dir_name in config.LABELS:
            train_idx = int(config.TRAIN_SPLIT * count)
            val_idx = int(config.VAL_SPLIT * count)
            test_idx = int(config.TEST_SPLIT * count)
            copy_images(config.IMAGE_ROOT / dir_name,
                        config.DATA_PATH / "train" / dir_name,
                        files[:train_idx])
            copy_images(config.IMAGE_ROOT / dir_name,
                        config.DATA_PATH / "validation" / dir_name,
                        files[train_idx:train_idx + val_idx])
            copy_images(config.IMAGE_ROOT / dir_name,
                        config.DATA_PATH / "test" / dir_name,
                        files[-test_idx:])


def main():
    first_dir = config.LABELS[0]
    if os.path.exists(config.DATA_PATH / "train" / first_dir):
        print("[INFO] Dataset is already ingested.")
        return

    ingest()
    print("[INFO] Raw dataset was successfully ingested.")

if __name__ == '__main__':
    main()
