import os
import random
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
import imageio as io

class DatasetLoad(Dataset):
    def __init__(self, cover_path: str, stego_path: str, transform: Tuple = None) -> None:
        self.cover_path = cover_path
        self.stego_path = stego_path
        self.transforms = transform

        self.cover_images = sorted(os.listdir(cover_path))
        self.stego_images = sorted(os.listdir(stego_path))

        # 确保cover和stego文件夹中的图片数量相同
        assert len(self.cover_images) == len(self.stego_images), "Cover and Stego images count should be the same"

    def __len__(self) -> int:
        return len(self.cover_images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        cover_img_name = self.cover_images[index]
        stego_img_name = self.stego_images[index]

        cover_img = io.imread(os.path.join(self.cover_path, cover_img_name))
        stego_img = io.imread(os.path.join(self.stego_path, stego_img_name))

        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)

        # 将标签和图像一起返回
        label_cover = torch.tensor(0, dtype=torch.long)
        label_stego = torch.tensor(1, dtype=torch.long)

        sample = {"cover": cover_img, "stego": stego_img, "label": [label_cover, label_stego]}
        return sample
