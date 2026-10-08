import logging
import os
from glob import glob
from itertools import chain
from multiprocessing import Pool
from typing import Any, Dict, List, Literal, Optional, Set

import h5py
import hdf5plugin
import numpy as np
import pytorch_lightning as pl
import skimage.util as util
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm


class CaliforniaDataModule(pl.LightningDataModule):
    def __init__(self, **hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        # Create folders sets
        val_fold = (self.hparams["test_set"] + 1) % 5

        self.train_transforms = []
        self.test_transforms = []

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.generator = torch.Generator().manual_seed(self.hparams["seed"])

        self.batch_size = self.hparams["batch_size"]

        self.assigned_folds = {
            "train": [
                x for x in range(5) if x not in (self.hparams["test_set"], val_fold)
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
            """
            self.train_dataset = HDF5CaliforniaDataset(
                self.root,
                transforms=self.train_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                mode=self.hparams["mode"],
                fold_list=self.assigned_folds["train"],
                attributes_filter=self.hparams["comments_filter"],
                debug=self.hparams["debug"],
                parallel_workers=self.hparams["parallel_workers"],
            )
            """
            self.train_dataset = PrePatchedDataset(
                self.root,
                transforms=self.train_transforms,
                mode=self.hparams["mode"],
                folds=self.assigned_folds["train"],
                attributes_filter=set(self.hparams["comments_filter"]),
            )

        if stage in ("fit", "validate", None):
            """
            self.val_dataset = HDF5CaliforniaDataset(
                self.root,
                transforms=self.test_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                mode=self.hparams["mode"],
                fold_list=self.assigned_folds["val"],
                attributes_filter=self.hparams["comments_filter"],
                debug=self.hparams["debug"],
                parallel_workers=self.hparams["parallel_workers"],
            )
            """
            self.val_dataset = PrePatchedDataset(
                self.root,
                transforms=self.test_transforms,
                mode=self.hparams["mode"],
                folds=self.assigned_folds["val"],
                attributes_filter=set(self.hparams["comments_filter"]),
            )

        if stage in ("test", "predict", None):
            """
            self.test_dataset = HDF5CaliforniaDataset(
                self.root,
                transforms=self.test_transforms,
                patch_size=self.hparams["patch_size"],
                keep_burned_only=self.hparams["keep_burned_only"],
                mode=self.hparams["mode"],
                fold_list=self.assigned_folds["test"],
                attributes_filter=self.hparams["comments_filter"],
                debug=self.hparams["debug"],
                parallel_workers=self.hparams["parallel_workers"],
            )
            """
            self.test_dataset = PrePatchedDataset(
                self.root,
                transforms=self.test_transforms,
                mode=self.hparams["mode"],
                folds=self.assigned_folds["test"],
                attributes_filter=set(self.hparams["comments_filter"]),
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


class PrePatchedDataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        mode: Literal["post", "prepost"],
        folds: Set[int],
        attributes_filter: Set[int],
        transforms=None,
    ) -> None:
        if mode not in ("post", "prepost"):
            raise ValueError(f"Invalid mode: {mode}")
        super().__init__()
        self.transforms = transforms
        self.final_transforms = ToTensor()

        self.post = []
        self.pre = []
        self.masks = []
        self.names = []
        print("Loading folds: ", folds)
        # Read hdf5 file and filter by fold and attributes
        with h5py.File(hdf5_file, "r") as f:
            for uuid, values in f.items():
                comments = set(values.attrs["comments"].tolist())
                if values.attrs["fold"] not in folds or comments & attributes_filter:
                    continue
                if mode == "prepost" and "pre_fire" not in values:
                    continue

                self.post.append(values["post_fire"][...])
                if mode == "prepost":
                    self.pre.append(values["pre_fire"][...])
                self.masks.append(values["mask"][...])
                self.names.append(uuid)

        # Convert to numpy arrays
        self.post = np.stack(self.post, axis=0, dtype=np.int32)
        self.pre = (
            np.stack(self.pre, axis=0, dtype=np.int32) if mode == "prepost" else None
        )
        self.masks = np.stack(self.masks, axis=0, dtype=np.int32)
        print("Normalizing data")
        # Normalize sentinel 2 data
        self.post = self.post / 10000
        if mode == "prepost":
            self.pre = self.pre / 10000
        print(f"Loaded {len(self)} patches")

    def __len__(self) -> int:
        return self.masks.shape[0]

    def __getitem__(self, index: int) -> Any:
        if self.transforms is not None:
            pass  # TODO: Implement other transforms
        result = {
            "name": self.names[index],
            "post": torch.from_numpy(self.post[index]).permute(2, 0, 1),
            "mask": torch.from_numpy(self.masks[index]).permute(2, 0, 1),
        }
        if self.pre is not None:
            result["pre"] = torch.from_numpy(self.pre[index]).permute(2, 0, 1)
        return result


class HDF5CaliforniaDataset(Dataset):
    def __init__(
        self,
        hdf5_folder: str,
        patch_size: int = 512,
        fold_list: List[int] = None,
        keep_burned_only: bool = True,
        transforms=None,
        mode: str = "post",
        pre_available: bool = False,
        attributes_filter: List[int] = [],
        debug: bool = False,
        parallel_workers: int = 4,
    ):
        # Assert validity
        if mode not in ["post", "prepost"]:
            raise ValueError("mode must be post or prepost")

        self.transforms = transforms
        self.patches = []
        # No folder list provided
        if fold_list is None or len(fold_list) == 0:
            fold_list = list(range(5))
        # Load all patches
        with Pool(parallel_workers) as p:
            self.patches = p.starmap(
                _get_patches,
                [
                    (
                        set(fold_list),
                        hdf5_file,
                        patch_size,
                        mode,
                        pre_available,
                        attributes_filter,
                        keep_burned_only,
                        debug,
                    )
                    for hdf5_file in glob(f"{hdf5_folder}/*.hdf5")
                ],
            )
        self.patches = np.concatenate(list(chain.from_iterable(self.patches)))
        print(f"Dataset len = {self.patches.shape[0]}")

    def __getitem__(self, item):
        result = {"image": self.patches[item, :, :, :-1]}
        if self.transforms is not None:
            for t in self.transforms:
                result = t(result)
        result["mask"] = torch.from_numpy(self.patches[item, :, :, -1]).unsqueeze(0)
        return result

    def __len__(self):
        return self.patches.shape[0]


def _get_patches(
    folds: Set[int],
    hdf5_file: str,
    patch_size: int,
    mode: str,
    pre_available: bool,
    attributes_filter: List[int],
    keep_burned_only: bool,
    debug: bool,
) -> List[NDArray]:
    # Assert validity
    if mode not in ["post", "prepost"]:
        raise ValueError("mode must be post or prepost")
    # Convert to string for consistency
    folds = set([str(x) for x in folds])
    # Load dataset
    with h5py.File(hdf5_file, "r") as dataset:
        patches = []
        elements = [
            dict(dataset[f"{f}/{k}"])
            for f in folds & set(dataset.keys())
            for k in dataset[f].keys()
        ]
        for sample in elements:
            # Filter
            comments = [
                int(c)
                for c in str(sample["post_fire"].attrs["comments"]).split("-")
                if c.isnumeric()
            ]
            if set(comments) & set(attributes_filter):
                continue
            if "pre_fire" not in sample and (pre_available or mode == "prepost"):
                continue
            if mode != "prepost":
                sample.pop("pre_fire", None)
            mask = sample.pop("mask")[...]
            # Init
            img = np.concatenate(
                list(sample.values()) + [mask.reshape(*mask.shape, 1)],
                axis=-1,
            )
            img_size = img.shape[0]
            usable_size = img_size // patch_size * patch_size
            overlapping_start = img_size - patch_size
            # Portioning
            to_cut = img[:usable_size, :usable_size]
            last_row = img[overlapping_start:, :usable_size]
            last_column = img[:usable_size, overlapping_start:]
            last_crop = img[overlapping_start:img_size, overlapping_start:img_size]
            # Crop
            wanted_crop_size = (patch_size, patch_size, img.shape[-1])
            last_row = util.view_as_blocks(last_row, wanted_crop_size)
            last_column = util.view_as_blocks(last_column, wanted_crop_size)
            crops = util.view_as_blocks(to_cut, wanted_crop_size)
            # Reshaping
            crops = crops.reshape(crops.shape[0] * crops.shape[1], *wanted_crop_size)
            last_row = last_row.reshape(last_row.shape[1], *wanted_crop_size)
            last_column = last_column.reshape(last_column.shape[0], *wanted_crop_size)
            last_crop = last_crop.reshape(1, *last_crop.shape)
            # Merge
            merged = np.concatenate([crops, last_column, last_row, last_crop])
            if keep_burned_only:
                merged = merged[merged[:, :, :, -1].sum(axis=(1, 2)) > 0]

            patches.append(merged)
            # Debug
            if debug and len(patches) > 2:
                break

    return patches


if __name__ == "__main__":
    dataset = PrePatchedDataset(
        "../../california_burned_areas/only_burned/burned_512x512.hdf5",
        "post",
        set(range(5)),
        [],
    )
