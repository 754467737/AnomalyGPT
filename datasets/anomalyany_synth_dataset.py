from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class AnomalyAnySynthDataset(Dataset):
    def __init__(self, run_dir: str, image_size: int = 224, mask_threshold: float = 0.5):
        self.run_dir = Path(run_dir)
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.samples: List[Tuple[Path, Path, int]] = []

        normal_dir = self.run_dir / "normal"
        for p in sorted(normal_dir.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                self.samples.append((p, None, 0))

        syn_img = self.run_dir / "synthetic" / "images"
        syn_mask = self.run_dir / "synthetic" / "masks"
        for p in sorted(syn_img.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                mp = syn_mask / f"{p.stem}.png"
                if not mp.exists():
                    mp = syn_mask / f"{p.stem}{p.suffix}"
                self.samples.append((p, mp, 1))

        self.img_t = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.mask_t = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p, label = self.samples[idx]
        image = Image.open(img_p).convert("RGB")
        image_t = self.img_t(image)
        if mask_p is None or not mask_p.exists():
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            m = Image.open(mask_p).convert("L")
            mask = self.mask_t(m)
            mask = (mask >= self.mask_threshold).float()
        return image_t, mask, torch.tensor(label, dtype=torch.float32), str(img_p)


class MVTecTestDataset(Dataset):
    def __init__(self, root: str, class_name: str, image_size: int = 224, mask_threshold: float = 0.5):
        root = Path(root) / class_name
        self.samples = []
        test_dir = root / "test"
        gt_dir = root / "ground_truth"
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect = defect_dir.name
            for p in sorted(defect_dir.iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                if defect == "good":
                    self.samples.append((p, None, 0))
                else:
                    mp = gt_dir / defect / f"{p.stem}_mask.png"
                    self.samples.append((p, mp, 1))

        self.img_t = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.mask_t = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.mask_threshold = mask_threshold
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p, label = self.samples[idx]
        image = self.img_t(Image.open(img_p).convert("RGB"))
        if mask_p is None or not mask_p.exists():
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            m = self.mask_t(Image.open(mask_p).convert("L"))
            mask = (m >= self.mask_threshold).float()
        return image, mask, torch.tensor(label, dtype=torch.float32), str(img_p)
