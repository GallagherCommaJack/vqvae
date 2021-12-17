from typing import Optional
import torch.utils.data as D
from PIL import Image
from pathlib import Path

import torchvision.transforms as TV


class Grayscale2RGB:
    def __init__(self):
        pass

    def __call__(self, img):
        if img.mode != "RGB":
            return img.convert("RGB")
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ImageFolder(D.Dataset):
    def __init__(
        self,
        root_dir: str,
        ix_file: str,
        res: int = 192,
        img_folder: Optional[str] = None,
    ):
        super().__init__()
        root_dir = Path(root_dir)
        self.img_root_dir = root_dir
        if img_folder:
            self.img_root_dir = self.img_root_dir / img_folder
        with open(root_dir / ix_file, "r") as f:
            self.ixs = [line[:-1] for line in f]
        self.transform = TV.Compose(
            [
                TV.RandomResizedCrop(res, scale=(0.75, 1), ratio=(1.0, 1.0)),
                Grayscale2RGB(),
                TV.ToTensor(),
                TV.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

    def __getitem__(self, ix):
        filename = self.img_root_dir / f"{self.ixs[ix]}.jpg"
        img = Image.open(filename)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.ixs)