import os
import random
import sys

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711))
    ])(pil_image)


class MDDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.size = opt.get("patch_size", 256)
        self.deg_types = opt["distortion"]

        self.distortion = {}
        for deg_type in self.deg_types:
            GT_paths = util.get_image_paths(
                opt["data_type"], os.path.join(opt["dataroot"], deg_type, 'GT')
            )
            LR_paths = util.get_image_paths(
                opt["data_type"], os.path.join(opt["dataroot"], deg_type, 'LQ')
            )
            self.distortion[deg_type] = (GT_paths, LR_paths)

        self.data_lens = [len(self.distortion[deg_type][0]) for deg_type in self.deg_types]

    @staticmethod
    def _align_pair(img_a, img_b):
        Ha, Wa = img_a.shape[:2]
        Hb, Wb = img_b.shape[:2]
        H0, W0 = min(Ha, Hb), min(Wa, Wb)
        return img_a[:H0, :W0, :], img_b[:H0, :W0, :]

    @staticmethod
    def _pad_to_min_size(img, min_h, min_w):
        H, W = img.shape[:2]
        pad_h = max(0, min_h - H)
        pad_w = max(0, min_w - W)
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        return img

    def __getitem__(self, index):
        type_id = int(index % len(self.deg_types))

        if self.opt["phase"] == "train":
            deg_type = self.deg_types[type_id]
            index = np.random.randint(self.data_lens[type_id])
        else:
            while index // len(self.deg_types) >= self.data_lens[type_id]:
                index += 1
                type_id = int(index % len(self.deg_types))
            deg_type = self.deg_types[type_id]
            index = index // len(self.deg_types)

        GT_path = self.distortion[deg_type][0][index]
        LQ_path = self.distortion[deg_type][1][index]

        img_GT = util.read_img(None, GT_path, None)  # float32 HWC BGR [0,1]
        img_LQ = util.read_img(None, LQ_path, None)

        img_GT, img_LQ = self._align_pair(img_GT, img_LQ)

        if self.opt["phase"] == "train":
            img_GT = self._pad_to_min_size(img_GT, self.size, self.size)
            img_LQ = self._pad_to_min_size(img_LQ, self.size, self.size)

            H, W = img_GT.shape[:2]  
            rnd_h = random.randint(0, H - self.size)
            rnd_w = random.randint(0, W - self.size)

            img_GT = img_GT[rnd_h:rnd_h + self.size, rnd_w:rnd_w + self.size, :]
            img_LQ = img_LQ[rnd_h:rnd_h + self.size, rnd_w:rnd_w + self.size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment(
                [img_LQ, img_GT],
                self.opt["use_flip"],
                self.opt["use_rot"],
                mode=self.opt["mode"],
            )

            if img_GT.shape[0] != self.size or img_GT.shape[1] != self.size:
                raise RuntimeError(f"Bad GT patch: {img_GT.shape}, GT_path={GT_path}, LQ_path={LQ_path}")
            if img_LQ.shape[0] != self.size or img_LQ.shape[1] != self.size:
                raise RuntimeError(f"Bad LQ patch: {img_LQ.shape}, GT_path={GT_path}, LQ_path={LQ_path}")

        else:
            H, W = img_GT.shape[:2]
            max_size = self.opt.get("val_max_size", self.size * 2)  

            if H > max_size or W > max_size:
                start_h = max(0, (H - max_size) // 2)
                start_w = max(0, (W - max_size) // 2)
                img_GT = img_GT[start_h:start_h + min(max_size, H), start_w:start_w + min(max_size, W), :]
                img_LQ = img_LQ[start_h:start_h + min(max_size, H), start_w:start_w + min(max_size, W), :]

            img_GT, img_LQ = self._align_pair(img_GT, img_LQ)

        # change color space if necessary
        if self.opt["color"]:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[0]
            img_LQ = util.channel_convert(img_LQ.shape[2], self.opt["color"], [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]

        lq4clip = clip_transform(img_LQ)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        return {"GT": img_GT, "LQ": img_LQ, "LQ_clip": lq4clip, "type": deg_type, "GT_path": GT_path}

    def __len__(self):
        return int(np.sum(self.data_lens))
