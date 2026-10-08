from itertools import chain
from typing import Any, Dict, Literal, Optional, Set

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class EuropeDataModule(pl.LightningDataModule):
    def __init__(self, **hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        available_folds = ["coral", "cyan", "grey", "lime", "magenta", "pink", "purple"]

        # Create folders sets
        test_set_index = available_folds.index(self.hparams["test_set"])
        val_set_index = (test_set_index + 1) % len(available_folds)
        val_fold = available_folds[val_set_index]

        self.train_transforms = []
        self.test_transforms = []

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.generator = torch.Generator().manual_seed(self.hparams["seed"])

        self.batch_size = self.hparams["batch_size"]

        self.assigned_folds = {
            "train": [
                x
                for x in available_folds
                if x not in (self.hparams["test_set"], val_fold)
            ],
            "val": [val_fold],
            "test": [self.hparams["test_set"]],
        }
        # Assert assigned_folds values are unique and not overlapping
        folds = list(chain(*self.assigned_folds.values()))
        assert len(set(folds)) == len(folds)

        self.root = self.hparams["root"]

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = EuropeDataset(
                self.root,
                transforms=self.train_transforms,
                mode=self.hparams["mode"],
                folds=self.assigned_folds["train"],
            )

        if stage in ("fit", "validate", None):
            self.val_dataset = EuropeDataset(
                self.root,
                transforms=self.test_transforms,
                mode=self.hparams["mode"],
                folds=self.assigned_folds["val"],
            )

        if stage in ("test", "predict", None):
            self.test_dataset = EuropeDataset(
                self.root,
                transforms=self.test_transforms,
                mode=self.hparams["mode"],
                folds=self.assigned_folds["test"],
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

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )


class EuropeDataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        mode: Literal["post", "prepost"],
        folds: Set[str],
        transforms=None,
    ) -> None:
        if mode not in ("post", "prepost"):
            raise ValueError(f"Invalid mode: {mode}")
        super().__init__()
        self.transforms = transforms

        self.post = []
        self.pre = []
        self.masks = []
        self.names = []
        print("Loading folds: ", folds)
        # Read hdf5 file and filter by fold and attributes
        with h5py.File(hdf5_file, "r") as f:
            for fold in folds:
                for uuid, values in f[fold].items():
                    if mode == "prepost" and "pre" not in values:
                        continue

                    self.post.append(values["post"][...])
                    if mode == "prepost":
                        self.pre.append(values["pre"][...])
                    self.masks.append(values["mask"][...])
                    self.names.append(uuid)

        # Convert to numpy arrays
        self.post = np.stack(self.post, axis=0, dtype=np.float32)
        self.pre = (
            np.stack(self.pre, axis=0, dtype=np.float32) if mode == "prepost" else None
        )
        self.masks = np.stack(self.masks, axis=0, dtype=np.float32).astype(
            dtype=np.int32
        )
        print(f"Loaded {len(self)} patches")

    def __len__(self) -> int:
        return self.masks.shape[0]

    def __getitem__(self, index: int) -> Any:
        if self.transforms is not None:
            pass  # TODO: Implement other transforms
        result = {
            "name": self.names[index],
            "post": torch.from_numpy(self.post[index]).permute(2, 0, 1) * 10,
            "mask": torch.from_numpy(self.masks[index]).permute(2, 0, 1),
        }
        # Clamp mask to 0-1
        result["mask"] = torch.clamp(result["mask"], 0, 1)
        if self.pre is not None:
            result["pre"] = torch.from_numpy(self.pre[index]).permute(2, 0, 1)
        return result


class LazyEuropeDataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        mode: Literal["post", "prepost"],
        folds: Set[str],
        transforms=None,
    ) -> None:
        if mode not in ("post", "prepost"):
            raise ValueError(f"Invalid mode: {mode}")
        super().__init__()
        self.transforms = transforms
        self.hdf5_file = hdf5_file

        self.post = []
        self.pre = []
        self.masks = []
        self.names = []
        print("Loading folds: ", folds)
        with h5py.File(hdf5_file, "r") as f:
            for fold in folds:
                for uuid in f[fold].keys():
                    self.post.append(f"{fold}/{uuid}/post")
                    self.masks.append(f"{fold}/{uuid}/mask")
                    self.names.append(uuid)

        self.post_sample = np.empty((512, 512, 12), dtype=np.float32)
        self.mask_sample = np.empty((512, 512, 1), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Any:
        if self.transforms is not None:
            pass  # TODO: Implement other transforms

        with h5py.File(self.hdf5_file, "r") as f:
            f[self.post[index]].read_direct(self.post_sample)
            f[self.masks[index]].read_direct(self.mask_sample)

        result = {
            "name": self.names[index],
            "post": torch.from_numpy(self.post_sample).clone().permute(2, 0, 1),
            "mask": torch.from_numpy(self.mask_sample).clone().permute(2, 0, 1),
        }
        return result
