from typing import Optional
import numpy as np
import torch.utils.data as D
import torch
from PIL import Image
from pathlib import Path

import torchvision.transforms as TV
import pytorch_lightning as pl


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
        try:
            filename = self.img_root_dir / f"{self.ixs[ix]}.jpg"
            img = Image.open(filename)
            img = self.transform(img)
            return img
        except:
            return None

    def __len__(self):
        return len(self.ixs)


def collate_skip_error(dataset):
    def f(batch):
        len_batch = len(batch)  # original batch length
        batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
        if len_batch > len(
            batch
        ):  # source all the required samples from the original dataset at random
            diff = len_batch - len(batch)
            for i in range(diff):
                batch.append(dataset[np.random.randint(0, len(dataset))])

        return D.dataloader.default_collate(batch)

    return f


class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        ix_file: str,
        res: int = 192,
        img_folder: Optional[str] = None,
        val_size=1,
        seed: int = 42,
        batch_size: int = 1,
        num_workers: int = 128,
    ):
        super().__init__()
        ds_base = ImageFolder(
            root_dir=root_dir, ix_file=ix_file, res=res, img_folder=img_folder
        )

        if isinstance(val_size, int):
            vsize = val_size * batch_size
        elif isinstance(val_size, float):
            vsize = (len(ds_base) * val_size // batch_size) * batch_size

        tsize = len(ds_base) - vsize

        dt, dv = D.random_split(
            ds_base, [tsize, vsize], generator=torch.Generator().manual_seed(seed)
        )
        self.dt = dt
        self.dv = dv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return D.DataLoader(
            self.dt,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return D.DataLoader(
            self.dv,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

