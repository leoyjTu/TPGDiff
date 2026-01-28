import os
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM',
    '.bmp', '.BMP', '.tif', '.TIF'
]

DEGRADATION_TYPES = ['blurry', 'hazy', 'low-light', 'noisy', 'rainy']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_paired_paths(dataroot):
    gt_paths, lq_paths, degradations = [], [], []
    for deg_type in DEGRADATION_TYPES:
        gt_dir = os.path.join(dataroot, deg_type, 'GT')
        lq_dir = os.path.join(dataroot, deg_type, 'LQ')

        paths_gt = _get_paths_from_images(gt_dir)
        paths_lq = _get_paths_from_images(lq_dir)

        if len(paths_gt) != len(paths_lq):
            print(f"[WARN] {deg_type}: GT({len(paths_gt)}) != LQ({len(paths_lq)}), please check pairing.")
        gt_paths.extend(paths_gt)
        lq_paths.extend(paths_lq)
        degradations.extend([deg_type] * len(paths_lq))

    return gt_paths, lq_paths, degradations


def generate_prior_csv(dataroot, mode='train'):
    split_root = os.path.join(dataroot, mode)
    gt_paths, lq_paths, degradations = get_paired_paths(split_root)

    records = {
        "lq_path": [],
        "gt_path": [],
        "deg_label": [],
    }

    for gt_image_path, lq_image_path, deg_label in tqdm(
        zip(gt_paths, lq_paths, degradations),
        total=len(gt_paths),
        desc=f"Building priors CSV ({mode})"
    ):
        try:
            _ = Image.open(gt_image_path).convert('RGB')
            _ = Image.open(lq_image_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to open images: GT={gt_image_path}, LQ={lq_image_path}, err={e}")
            continue

        records["lq_path"].append(lq_image_path)
        records["gt_path"].append(gt_image_path)
        records["deg_label"].append(deg_label)

    out_csv = os.path.join(dataroot, f"priors_{mode}.csv")
    pd.DataFrame.from_dict(records).to_csv(
        out_csv,
        index=False,
        sep="\t"
    )


if __name__ == "__main__":
    #   datasets/
    #       ├── train/
    #       │     ├── blurry/...
    #       │     ├── hazy/...
    #       └── val/
    #             ├── blurry/...
    #             ├── hazy/...
    dataroot = 'datasets/universal'

    generate_prior_csv(dataroot, 'train')
    generate_prior_csv(dataroot, 'val')
