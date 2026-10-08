import argparse
import logging
import os
import pathlib
from pathlib import Path

import comet_ml
import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
import utils
from hydra.utils import instantiate
from lightning_modules import (
    CaliforniaDataModule,
    EuropeDataModule,
    IndonesiaDataModule,
)
from neural_net import MagnifierNet
from omegaconf import DictConfig
from pytorch_lightning.tuner import Tuner

DATAMODULES = {
    "california": CaliforniaDataModule,
    "europe": EuropeDataModule,
    "indonesia": IndonesiaDataModule,
}


def build_datamodule(cfg: DictConfig):
    name = cfg["dataset"].get("name", "europe")
    if name not in DATAMODULES:
        raise ValueError(
            f"Unknown dataset '{name}'. Expected one of {list(DATAMODULES)}."
        )
    return DATAMODULES[name](**cfg["dataset"])


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # Set common seed
    pl.seed_everything(47, True)
    # Create model
    pl_model = instantiate(cfg["model"])
    # Setup logger
    logger = loggers.CometLogger(
        **cfg["logger"], experiment_name=pl_model.__class__.__name__
    )
    # Setup checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        filename="{epoch}-{val_loss:.2f}",
        every_n_epochs=11,
    )
    # Setup early stopping callback
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, mode="min"
    )
    # Setup learning rate logger
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # Setup model summary caallback
    model_summary_callback = pl.callbacks.RichModelSummary()
    # Set learning rate finder
    lr_finder = pl.callbacks.LearningRateFinder(max_lr=0.001)
    # Setup trainer
    trainer = pl.Trainer(
        **cfg["trainer"],
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            lr_logger,
            model_summary_callback,
            pl.callbacks.RichProgressBar(),
            # lr_finder,
        ],
    )

    if "lr_find" == cfg.mode:
        tuner = Tuner(trainer)
        suggestions = []
        for i in range(5):
            cfg["dataset"]["test_set"] = i
            lr_finder = tuner.lr_find(pl_model, datamodule=build_datamodule(cfg))
            suggestions.append(lr_finder.suggestion())
        print(
            "All suggestions: ",
            suggestions,
            " Average: ",
            sum(suggestions) / len(suggestions),
        )
        return

    # Create datamodule
    datamodule = build_datamodule(cfg)
    if "train" == cfg.mode:
        trainer.fit(pl_model, datamodule=datamodule)
        logger.experiment.log_model(
            "Best model", trainer.checkpoint_callback.best_model_path
        )
        trainer.test(pl_model, datamodule=datamodule, ckpt_path="best")
    if "test" == cfg.mode:
        trainer.test(pl_model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    if "predict" == cfg.mode:
        trainer.predict(pl_model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    logger.experiment.log_parameter("model", cfg.model_name)
    logger.experiment.end()


if __name__ == "__main__":
    main()
