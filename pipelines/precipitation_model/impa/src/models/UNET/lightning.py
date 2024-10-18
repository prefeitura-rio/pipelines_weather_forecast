# -*- coding: utf-8 -*-
import torch

from pipelines.precipitation_model.impa.src.models.lightning_module import LModule
from pipelines.precipitation_model.impa.src.models.UNET.unet_parts import (
    DoubleConv,
    Down,
    OutConv,
    Up,
)


class model(LModule):
    def __init__(
        self,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0,
        truth: torch.Tensor = None,
        context: tuple = None,
        truth_val: torch.Tensor = None,
        context_val: tuple = None,
        bilinear: bool = True,
        loss: int = 1,
        n_before: int = 10,
        n_after: int = 20,
        normalized: int = 0,
        xmax: float = 3 * 20 / 5,
        x_mean: float = 0,
        x_std: float = 1,
        data_modification: list = None,
        dimension_division: float = None,
        merge: bool = False,
        satellite: bool = False,
        **kwargs,
    ):
        super().__init__(
            truth=truth,
            context=context,
            truth_val=truth_val,
            context_val=context_val,
            n_before=n_before,
            n_after=n_after,
            normalized=normalized,
            loss=loss,
            xmax=xmax,
            x_mean=x_mean,
            x_std=x_std,
            data_modification=data_modification,
            merge=merge,
            satellite=satellite,
        )
        self.save_hyperparameters()

        self.lr = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay

        self.bilinear = self.hparams.bilinear
        self.weighted = self.hparams.weights
        self.dimension_division = self.hparams.dimension_division
        self.dim = 64

        if self.dimension_division is not None:
            self.dim /= self.dimension_division

        self.dim = int(self.dim)

        self.inc = DoubleConv(self.channels_in, self.dim)
        self.down1 = Down(self.dim, self.dim * 2)
        self.down2 = Down(self.dim * 2, self.dim * 4)
        self.down3 = Down(self.dim * 4, self.dim * 8)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(self.dim * 8, self.dim * 16 // factor)
        self.up1 = Up(self.dim * 16, self.dim * 8 // factor, self.bilinear)
        self.up2 = Up(self.dim * 8, self.dim * 4 // factor, self.bilinear)
        self.up3 = Up(self.dim * 4, self.dim * 2 // factor, self.bilinear)
        self.up4 = Up(self.dim * 2, self.dim, self.bilinear)
        self.outc = OutConv(self.dim, self.channels_out)

    def forward(self, x):
        # torch.flip(x,dims=[1])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
