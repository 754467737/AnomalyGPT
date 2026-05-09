from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXT]


def parse_defect_from_filename(path: Path) -> str:
    name = path.stem
    if "_" not in name:
        return "anomalous"
    defect = name.split("_", 1)[1].replace("_", " ").strip()
    return defect if defect else "anomalous"


class AnomalyAnySynthDataset(Dataset):
    def __init__(self, data_root: str, obj_name: Optional[str], seed: str, image_size: int = 224, mask_threshold: float = 0.5):
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.samples: List[Tuple[Path, Optional[Path], int, str, str]] = []

        root = Path(data_root)
        objs = [obj_name] if obj_name else sorted([p.name for p in root.iterdir() if p.is_dir()])

        for obj in objs:
            base = root / obj
            normal_dir = base / "normal"
            syn_img = base / "synthetic" / seed / "images"
            syn_mask = base / "synthetic" / seed / "masks"

            for p in _list_images(normal_dir):
                self.samples.append((p, None, 0, obj, "normal"))

            for p in _list_images(syn_img):
                mp = syn_mask / f"{p.stem}.png"
                if not mp.exists():
                    mp = syn_mask / f"{p.stem}{p.suffix}"
                self.samples.append((p, mp, 1, obj, parse_defect_from_filename(p)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for data_root={data_root}, obj={obj_name}, seed={seed}")

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
        img_p, mask_p, label, obj, defect = self.samples[idx]
        image_t = self.img_t(Image.open(img_p).convert("RGB"))
        if mask_p is None or (not mask_p.exists()):
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            mask = self.mask_t(Image.open(mask_p).convert("L"))
            mask = (mask >= self.mask_threshold).float()
        return image_t, mask, torch.tensor(label, dtype=torch.float32), str(img_p), obj, defect


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
            for p in _list_images(defect_dir):
                if defect == "good":
                    self.samples.append((p, None, 0))
                else:
                    self.samples.append((p, gt_dir / defect / f"{p.stem}_mask.png", 1))

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
