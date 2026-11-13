"""PyTorch Dataset for one split.

    from donkey_seg_datagen.dataset import DonkeySegDataset
    ds = DonkeySegDataset("out", "train")
    img, mask = ds[0]      # img: float32 3xHxW in [0,1]; mask: int64 HxW in {0,1}
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .index import load_index


class DonkeySegDataset(Dataset):
    def __init__(self, root, split: str, transform=None, return_meta: bool = False):
        """transform: callable(img_tensor, mask_tensor) -> (img, mask), applied to both together."""
        self.items = load_index(root, split)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        r = self.items[i]
        img = torch.from_numpy(np.asarray(Image.open(r["image_path"]).convert("RGB")).copy())
        img = img.permute(2, 0, 1).float().div_(255.0)
        mask = torch.from_numpy(np.asarray(Image.open(r["mask_path"])).astype(np.int64))
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return (img, mask, r) if self.return_meta else (img, mask)
