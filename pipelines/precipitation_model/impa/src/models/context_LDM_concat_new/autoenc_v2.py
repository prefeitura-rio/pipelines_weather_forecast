# -*- coding: utf-8 -*-
import numpy as np
import pytorch_lightning as pl
import torch

from pipelines.precipitation_model.impa.src.models.context_LDM_concat_new.autoencoder.distributions import (
    DiagonalGaussianDistribution,
)
from pipelines.precipitation_model.impa.src.models.context_LDM_concat_new.autoencoder.loss import (
    LPIPSWithDiscriminator,
)
from pipelines.precipitation_model.impa.src.models.context_LDM_concat_new.autoencoder.modules import (
    Decoder,
    Encoder,
)


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        learning_rate=0.00002,
        n_before=10,
        n_after=20,
        img_size=192,
        reduc_factor=4,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        z_channels = 4
        double_z = True
        self.image_key = image_key
        self.n_before = n_before
        ch_mult = [2**i for i in range(int(np.log2(reduc_factor)) + 1)]
        self.channels = 128
        self.encoder = Encoder(
            double_z=double_z,
            z_channels=z_channels,
            resolution=img_size,
            in_channels=2 * n_before,
            ch=self.channels,
            ch_mult=ch_mult,  # num_down = len(ch_mult)-1
            num_res_blocks=2,
            attn_resolutions=[4],
            dropout=0.2,
        )
        self.decoder = Decoder(
            double_z=double_z,
            z_channels=z_channels,
            resolution=img_size,
            in_channels=2 * n_before,
            out_ch=n_after,
            ch=self.channels,
            ch_mult=ch_mult,  # num_down = len(ch_mult)-1
            num_res_blocks=2,
            attn_resolutions=[4],
            dropout=0.2,
        )
        self.loss = LPIPSWithDiscriminator(
            disc_start=50001,
            kl_weight=0.00001,
            disc_weight=0.5,
            disc_in_channels=n_before,
        )
        assert double_z
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
