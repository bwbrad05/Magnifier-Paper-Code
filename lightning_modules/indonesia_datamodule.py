from typing import Literal

import polars as pl
import xarray as xr
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class IndonesiaDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        test_set: Literal[0, 1, 2, 3, 4],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.train_dataset = IndonesiaDataset(
            "data/indonesia", "train", splits=[i for i in range(5) if i != test_set]
        )
        self.val_dataset = IndonesiaDataset("data/indonesia", "val", splits=[test_set])
        self.test_dataset = IndonesiaDataset(
            "data/indonesia", "test", splits=[test_set]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )


class IndonesiaDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transforms=None,
        splits: list[int] = [1],  # 1 era il test, abbiamo fatto sampling dal zero
    ):
        self.transform = transforms
        self.samples = pl.read_parquet(f"{root}/splits.parquet").filter(
            pl.col("fold").is_in(splits)
        )
        self.samples = self.samples.select(
            image=root + "/images/" + pl.col("files") + ".tif",
            mask=root + "/masks/" + pl.col("files") + "_mask.tif",
        ).sort("image")

    def __getitem__(self, index: int) -> dict:
        i_path, m_path = self.samples.row(index)
        image = xr.open_dataarray(i_path).fillna(0).to_numpy()
        mask = xr.open_dataarray(m_path).fillna(0).to_numpy()
        return {"post": image, "mask": mask}

    def __len__(self) -> int:
        return len(self.samples)
