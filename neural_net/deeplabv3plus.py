import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics as tm
import torchvision.utils as vutils
import utils
from loss import AsymmetricUnifiedFocalLoss
from torch.nn import functional as F


class DeepLabV3Plus(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str,
        n_channels: int = 12,
        n_classes: int = 2,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=n_channels,
            classes=n_classes,
        )
        self.loss = AsymmetricUnifiedFocalLoss(0.5, 0.6, 0.1)
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

    def forward(self, x):
        return self.model(x)

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
        out = self(images)
        out = torch.softmax(out, dim=1)
        loss = self.loss(out, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        out = self(images)
        out = torch.softmax(out, dim=1)
        loss = self.loss(out, masks)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        out = self(images)
        out = torch.softmax(out, dim=1)
        loss = self.loss(out, masks)

        self.test_metrics.update(out, masks)
        if batch_idx in self.batch_to_log:
            self._log_images(
                images,
                masks,
                out.argmax(dim=1),
                out,
                prefix="test",
                draw_rgb=True,
                draw_heatmap=True,
            )

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def _log_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        pred_mask: torch.Tensor,
        logits: torch.Tensor,
        prefix: str = "",
        draw_rgb: bool = False,
        draw_heatmap: bool = False,
    ):
        for i, figure in enumerate(utils.draw_figure(masks, pred_mask)):
            if hasattr(self.logger.experiment, "log_image"):
                self.logger.experiment.log_figure(
                    figure=figure,
                    figure_name=f"{prefix}S{self.global_step}N{i}",
                    step=self.global_step,
                )
