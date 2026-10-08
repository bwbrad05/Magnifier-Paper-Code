# Copyright 2022 Daniele Rege Cambrin
from itertools import chain, groupby, product
from random import random
from typing import Dict, List, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
import torchvision.transforms.functional as TF
import utils
from loss import AsymmetricUnifiedFocalLoss
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from transformers import SegformerConfig, SegformerDecodeHead, SegformerModel


class MagnifierNet(pl.LightningModule):
    """
    A multiple net for patch of different size.

    Parameters
    ----------
    small_patch_size: int, optional
        The size of the patch to be considered as small
    num_classes: int, optional
        the number of classes
    channels: int, optional
        the number of channels
    big_patch_size: int, optional
        the size of the patch to be considered as big
    model: str, optional
        the model to use, can be "segformer" or "unet"
    """

    def __init__(
        self,
        model: Literal["segformer", "unet", "deeplabv3plus"] = "segformer",
        big_patch_size: int = 512,
        small_patch_size: int = 64,
        num_classes: int = 2,
        channels: int = 12,
        learning_rate: float = 0.003,
        encoder_name: str = "resnet18",
    ):
        super().__init__()
        if model not in ["segformer", "unet", "deeplabv3plus"]:
            raise ValueError("Model not supported")
        # Nets
        self.big_net = None
        self.small_net = None
        self.head = None
        self.postprocess = lambda x: x
        self.backbone_forward = lambda backbone, x: backbone(x)

        if model == "segformer":
            self._segformer_init(num_classes, channels, big_patch_size)
        elif model == "unet":
            self._unet_init(num_classes, channels, encoder_name)
        elif model == "deeplabv3plus":
            self._deeplabv3plus_init(num_classes, channels, encoder_name)
        # Parameters init
        self.small_patch_size = small_patch_size
        # Loss
        self.loss = AsymmetricUnifiedFocalLoss(0.5, 0.6, 0.1)
        # Metrics
        self.test_metrics = tm.MetricCollection(
            {
                "IoU": tm.ClasswiseWrapper(
                    tm.JaccardIndex("multiclass", num_classes=2, average="none")
                ),
                "F1Score": tm.ClasswiseWrapper(
                    tm.F1Score(
                        "multiclass",
                        num_classes=2,
                        average="none",
                        multidim_average="global",
                    )
                ),
            }
        )

        self.batch_to_log = [0, 5]
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor):
        # Forward all images to the net
        hidden_states = list(self.backbone_forward(self.big_net, x))
        # Create crops and their positions
        cropped_batch = []
        positions = []
        for image in x:
            imgs, pos = utils.crop_image(image, self.small_patch_size)
            cropped_batch.append(imgs)
            positions.append(pos)
        positions = list(chain(*positions))
        cropped_batch = torch.concat(cropped_batch)
        num_crops = (x.size()[-1] // self.small_patch_size) ** 2
        # Forward to backbone
        crops_out = self.backbone_forward(self.small_net, cropped_batch)
        # Recompose hidden states
        for i, hidden in enumerate(crops_out):
            hidden_states[i] = torch.concat(
                [
                    hidden_states[i],
                    torch.concat(
                        [
                            utils.recompose_image(
                                hidden[i : i + num_crops],
                                positions[i : i + num_crops],
                            ).unsqueeze(0)
                            for i in range(0, hidden.size()[0], num_crops)
                        ]
                    ),
                ],
                dim=1,
            )

        try:
            out = self.head(hidden_states)
        except (TypeError, IndexError, AttributeError):
            out = self.head(*hidden_states)

        return self.postprocess(out)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=55, power=1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        # Forward
        out = self(images)
        out = torch.softmax(out, dim=1)
        # Loss
        loss = self.loss(out, masks)
        # If loss is nan, stop training
        if torch.isnan(loss):
            return None
        # Log
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        # Forward
        out = self(images)
        out = torch.softmax(out, dim=1)
        # Loss
        loss = self.loss(out, masks)
        # Log
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        out = self(images)
        out = torch.softmax(out, dim=1)

        self.test_metrics.update(out, masks)

        if batch_idx in self.batch_to_log:
            for i, figure in enumerate(utils.draw_figure(masks, out.argmax(dim=1))):
                self.logger.experiment.log_figure(
                    figure=figure,
                    figure_name=f"testS{self.global_step}N{i}",
                    step=self.global_step,
                )

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def _segformer_init(
        self, num_classes: int = 2, channels: int = 12, big_patch_size: int = 512
    ):
        config = SegformerConfig(
            num_channels=channels,
            num_labels=num_classes,
            semantic_loss_ignore_index=-100,
        )

        self.big_net = SegformerModel(config)
        self.small_net = SegformerModel(config)

        config.hidden_sizes = [h * 2 for h in config.hidden_sizes]
        self.head = SegformerDecodeHead(config)

        self.postprocess = lambda x: nn.functional.interpolate(
            x, size=big_patch_size, mode="bilinear", align_corners=False
        )

        self.backbone_forward = lambda backbone, x: backbone(
            x, output_hidden_states=True
        )[1]

    def _unet_init(
        self, num_classes: int = 2, channels: int = 12, encoder: str = "resnet18"
    ):
        self.big_net = get_encoder(encoder, in_channels=channels)
        self.small_net = get_encoder(encoder, in_channels=channels)

        decoder_channels = (256, 128, 64, 32, 16)
        self.head = UnetDecoder(
            encoder_channels=[c * 2 for c in self.small_net.out_channels],
            decoder_channels=decoder_channels,
        )
        self.postprocess = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
            activation=None,
        )

    def _deeplabv3plus_init(
        self, num_classes: int = 2, channels: int = 12, encoder: str = "resnet18"
    ):
        self.big_net = get_encoder(encoder, in_channels=channels, output_stride=16)
        self.small_net = get_encoder(encoder, in_channels=channels, output_stride=16)

        decoder = DeepLabV3PlusDecoder(
            encoder_channels=[c * 2 for c in self.small_net.out_channels],
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )

        head = SegmentationHead(
            in_channels=decoder.out_channels,
            out_channels=num_classes,
            kernel_size=1,
            activation=None,
            upsampling=4,
        )

        self.head = decoder

        self.postprocess = head
