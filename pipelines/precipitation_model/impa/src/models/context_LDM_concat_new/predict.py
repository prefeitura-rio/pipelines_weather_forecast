# -*- coding: utf-8 -*-
import pathlib
from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pipelines.precipitation_model.impa.src.data.HDFDatasetLocations import (
    HDFDatasetLocations,
)
from pipelines.precipitation_model.impa.src.models.context_LDM_concat_new.lightning_v2 import (
    Diffusion_Model,
)
from pipelines.precipitation_model.impa.src.utils.general_utils import print_ok
from pipelines.precipitation_model.impa.src.utils.hdf_utils import array_to_pred_hdf

MEAN = 0.08
STD = 0.39


def transform2(X, mean, std, std_fac):
    X = torch.log1p(X.nan_to_num(0.0))
    X_norm = (X - mean) / (std_fac * std)
    X_norm[X_norm > 1] = 1
    X_norm[X_norm < -1] = -1
    return X_norm


def inv_transform(X, mean, std, std_fac):
    return torch.expm1(torch.permute(X, (0, 2, 3, 1)) * std_fac * std + mean)


def main(args_dict, parameters_dict):
    autoencoder_path = (
        "pipelines/precipitation_model/impa/src/models/context_LDM_VAE/"
        + args_dict["vae_hash"]
        + "/train/"
        + args_dict["dataframe"]
        + "/model_train.pt"
    )

    input_model_filepath = args_dict["input_model_filepath"]
    locations = args_dict["locations"]
    dataframe_filepath = args_dict["dataframe_filepath"]
    output_predict_filepaths = args_dict["output_predict_filepaths"]

    batch_size = args_dict["batch_to_predict"]
    n_predict = 1
    n_after = args_dict["n_after"]
    n_before = args_dict["n_before"]
    if args_dict["data_modification"] is not None:
        leadtime_conditioning = "Lead_time_cond" in args_dict["data_modification"]
    else:
        leadtime_conditioning = False

    torch.set_float32_matmul_precision("medium")
    parameters_dict_copy = parameters_dict.copy()
    for flag in ["n_epoch", "std_fac", "weight_bool", "model_name", "vae_hash"]:
        del parameters_dict_copy[flag]

    if (
        pathlib.Path(input_model_filepath).suffix == ".pt"
        or pathlib.Path(input_model_filepath).suffix == ".joblib"
    ):
        model = Diffusion_Model(autoenc_kl=autoencoder_path, **parameters_dict_copy)
        model.load_state_dict(torch.load(input_model_filepath, map_location="cuda:0"), strict=False)
    elif pathlib.Path(input_model_filepath).suffix == ".ckpt":
        model = Diffusion_Model.load_from_checkpoint(
            input_model_filepath, map_location="cuda:0"
        )  # , **parameters_dict, context=ds[0][0], truth=ds[0][1])
    else:
        raise ValueError("Invalid model file extension.")
    model.eval()

    assert locations is not None, "Locations must be provided."

    for output_predict_filepath in output_predict_filepaths:
        output_predict_filepath = pathlib.Path(output_predict_filepath)
        if output_predict_filepath.is_file():
            if args_dict["overwrite"]:
                output_predict_filepath.unlink()
            else:
                raise FileExistsError(
                    "Output file already exists. Pass 'overwrite' as true to overwrite."
                )

    # Load data
    for i, location in enumerate(locations):
        ds = HDFDatasetLocations(
            dataframe_filepath,
            [location],
            n_after=n_after,
            n_before=n_before,
            leadtime_conditioning=leadtime_conditioning,
        )
        ds.x_transform = partial(transform2, mean=MEAN, std=STD, std_fac=args_dict["std_fac"])
        ds.y_transform = partial(transform2, mean=MEAN, std=STD, std_fac=args_dict["std_fac"])
        test_dataloader = DataLoader(
            ds, batch_size=batch_size, num_workers=args_dict["num_workers"]
        )

        s2 = len(ds.keys)
        ni = ds[0][0].shape[1]
        array_pre = torch.zeros((n_predict, s2, n_after, ni, ni))
        for i in range(n_predict):
            predictions = pl.Trainer(devices=[0], logger=False, enable_checkpointing=False).predict(
                model, test_dataloader
            )
            predictions = torch.cat(predictions, axis=0)
            predictions = predictions.squeeze(1)
            array_pre[i, :, :, :, :] = predictions

        predictions = torch.mean(array_pre, 0)
        predictions = inv_transform(predictions, mean=MEAN, std=STD, std_fac=args_dict["std_fac"])
        array_to_pred_hdf(predictions, ds.keys, ds.future_keys, output_predict_filepaths[i])

    ok_message = "OK: Saved predictions successfully."
    print_ok(ok_message)
