# -*- coding: utf-8 -*-
# flake8: noqa: E203

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningModule
from tqdm import tqdm

from pipelines.precipitation_model.impa.src.models.context_LDM_concat_new.autoenc_v2 import (
    AutoencoderKL,
)

# from pipelines.precipitation_model.impa.src.models.context_LDM_concat.autoencoder.autoenc_old import AutoencoderKL
from pipelines.precipitation_model.impa.src.models.context_LDM_concat_new.ddim import (
    DDIMSampler,
)
from pipelines.precipitation_model.impa.src.models.context_LDM_concat_new.model import (
    get_named_beta_schedule,
    linear_beta_schedule,
)
from pipelines.precipitation_model.impa.src.models.context_LDM_concat_new.openaimodel import (
    UNetModel,
)
from pipelines.precipitation_model.impa.src.utils.data_utils import (
    data_modification_options,
)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class Diffusion_Model(LightningModule):
    def __init__(
        self,
        autoenc_kl,
        learning_rate: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        image_size: int = 256,
        timesteps: int = 1000,
        n_after: int = 10,
        n_before: int = 20,
        scheduler: str = "linear",
        context: torch.Tensor = None,
        truth: torch.Tensor = None,
        elev: torch.Tensor = None,
        context_val: torch.Tensor = None,
        truth_val: torch.Tensor = None,
        elev_val: torch.Tensor = None,
        x_mean: float = 0,
        x_std: float = 1,
        sampler: str = "ddim",
        max_linsc: float = 0.01,
        data_modification=None,
        normalized: int = 0,
        std_fac: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.flip = False
        if context is not None:
            self.fixed_context = context[None, :, :, :]
        if truth is not None:
            self.ground_truth = truth[None, :, :, :]
        if context_val is not None:
            self.fixed_context_val = context_val[None, :, :, :]
        if truth_val is not None:
            self.ground_truth_val = truth_val[None, :, :, :]
        if elev is not None:
            self.elev = elev[None, :, :, :]
        if elev_val is not None:
            self.elev_val = elev_val[None, :, :, :]

        self.n_channels = n_before
        self.data_modification = data_modification
        self.dm_option = data_modification_options
        if data_modification is not None:
            for _, value in enumerate(self.data_modification):
                self.dm_option[value] = True

        self.old = True
        self.sampler = sampler
        self.normalized = normalized
        self.std_fac = std_fac
        self.max = max_linsc
        self.show_im = 1
        self.x_mean = x_mean
        self.x_std = x_std

        self.val_loss_steps = []
        self.b1 = b1
        self.b2 = b2
        self.learning_rate = learning_rate
        self.n_before = n_before
        self.n_after = n_after

        self.timesteps = timesteps
        self.num_timesteps = timesteps
        self.scheduler = scheduler
        self.batch_train = None
        self.batch_val = None
        self.cond = True

        self.use_elev = self.dm_option["Elevation"]
        self.use_date = self.dm_option["Hour_data"]
        self.use_latlon = self.dm_option["Lat_lon"]
        self.use_leadtime = self.dm_option["Lead_time_cond"]
        self.no_context_train = self.dm_option["No_context"]
        self.predict_context = self.dm_option["Pred_context"]

        # define model
        self.autoencoder_obs = AutoencoderKL(
            embed_dim=n_before, n_before=n_before, n_after=n_after
        ).requires_grad_(False)

        self.autoencoder_obs.load_state_dict(
            torch.load(autoenc_kl, map_location="cuda:0"), strict=False
        )

        self.channels = self.autoencoder_obs.embed_dim
        self.down_factor = self.autoencoder_obs.encoder.in_ch_mult[-1]
        self.image_size = image_size // self.down_factor

        self.model = UNetModel(
            image_size=self.image_size,
            in_channels=2 * self.channels,
            out_channels=self.channels,
            model_channels=32 * 3,
            num_res_blocks=3,
            attention_resolutions=(4, 2, 1),
            num_heads=3,
            dims=2,
            dropout=0.1,
        )

        if self.scheduler == "cosine":
            self.register_buffer("betas", get_named_beta_schedule(self.scheduler, self.timesteps))
        else:
            self.register_buffer("betas", linear_beta_schedule(self.timesteps, self.max))

        # define alphas
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, axis=0))
        self.register_buffer(
            "alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))

        # calculations forward diffusion q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

        # calculations x0 conditional backward q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )

        self.ddim_sampler = DDIMSampler(self, timesteps=self.timesteps)

    # define basic backward sampling (single timestep)
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, cond=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # define backward sampling loop (all timesteps)
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond=None):
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=self.device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):
            img = self.p_sample(
                model,
                img,
                torch.full((b,), i, device=self.device, dtype=torch.long),
                i,
                cond=cond,
            )
            imgs.append(img.cpu().numpy())
        return imgs

    # do backward sampling
    @torch.no_grad()
    def sample(self, model, image_size, batch_size, channels, cond=None):
        ret = self.p_sample_loop(
            model, shape=(batch_size, channels, image_size, image_size), cond=cond
        )
        return torch.Tensor(np.array(ret))

    def predict_step(self, batch, batch_idx):
        x_before, _, metadata = batch
        n_after = _.shape[3]
        x = rearrange(x_before, "b h w c -> b c h w")
        x_small = x[:, : 2 * self.n_before : 2]
        x_large = x[:, 1 : 2 * self.n_before + 1 : 2]
        x_before = torch.cat([x_small, x_large], dim=1)

        x_before = self.autoencoder_obs.encode(x_before).sample()

        if self.sampler == "ddim":
            unconditional_conditioning = torch.full(x_before.shape, -0.1)
            shape = (self.channels, self.image_size, self.image_size)
            samples, _ = self.ddim_sampler.sample(
                shape=shape,
                S=self.timesteps // 10,
                batch_size=x_before.shape[0],
                conditioning=x_before,
                verbose=False,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=0.8,
            )
        elif self.sampler == "ddpm":
            samples = self.sample(
                self.model,
                image_size=self.image_size,
                batch_size=x_before.shape[0],
                channels=self.channels,
                cond=x_before,
            )[-1]
        else:
            raise NotImplementedError()

        # print(f"PREDICTION-> min:{samples.min()},max:{samples.max()}")
        pred = self.autoencoder_obs.decode(samples)
        if self.predict_context:
            pred = rearrange(pred, "b (r c) h w -> b h w c r", c=n_after)
            return pred[:, :, :, :, 0]
        else:
            return pred
