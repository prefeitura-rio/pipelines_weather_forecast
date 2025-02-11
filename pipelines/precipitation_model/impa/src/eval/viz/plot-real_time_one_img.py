# -*- coding: utf-8 -*-
# flake8: noqa: E501

"""
Keep only lag 18. This lag represents 3h from last observation.
"""
import datetime
import json
import pathlib
from argparse import ArgumentParser
from multiprocessing.pool import Pool

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from prefeitura_rio.pipelines_utils.logging import log

from pipelines.precipitation_model.impa.src.eval.metrics.metrics import metrics_dict
from pipelines.precipitation_model.impa.src.utils.eval_utils import get_img
from pipelines.precipitation_model.impa.src.utils.general_utils import print_warning
from pipelines.precipitation_model.impa.src.utils.hdf_utils import get_dataset_keys

dataset_dict = {
    "SAT": {
        "ground_truth_file": "pipelines/precipitation_model/impa/data/dataframes/SAT-CORRECTED-ABI-L2-RRQPEF-real_time-rio_de_janeiro/test.hdf",
        "grid_file": "pipelines/precipitation_model/impa/data/dataframe_grids/rio_de_janeiro-res=2km-256x256.npy",
    },
    "MDN": {
        "ground_truth_file": "pipelines/precipitation_model/impa/data/dataframes/MDN-d2CMAX-DBZH-real_time/test.hdf",
        "grid_file": "pipelines/precipitation_model/impa/data/dataframe_grids/rio_de_janeiro-res=700m-256x256.npy",
    },
}

parser = ArgumentParser()
parser.add_argument("dataset", type=str, choices=["SAT", "MDN"])
parser.add_argument("--num_workers", type=int, default=16)
args = parser.parse_args()

NLAGS = 18
BG_COLOR = "white"
METRICS_NAMES = ["log-MAE", "log-MSE", "CSI1", "CSI8"]
order = np.array([[1, 1, -1, -1]]).reshape(1, -1)

HEIGHT = 500
WIDTH = 500

config = pathlib.Path(
    f"pipelines/precipitation_model/impa/src/eval/real_time_config_{args.dataset}.json"
)
with open(config, "r") as json_file:
    specs_dict = json.load(json_file)

ground_truth_df = h5py.File(dataset_dict[args.dataset]["ground_truth_file"])
latlons = np.load(dataset_dict[args.dataset]["grid_file"])
feature = ground_truth_df["what"].attrs["feature"]
timestep = int(ground_truth_df["what"].attrs["timestep"])

keys = get_dataset_keys(ground_truth_df)
last_obs = keys[-1]
last_obs_dt = pd.to_datetime(last_obs)

log(f"last_obs: {last_obs}, last_obs_dt: {last_obs_dt}")

preds = [ground_truth_df]
log(f"preds: {preds}")
model_names = ["Ground truth"]
for model_name in specs_dict["models"].keys():
    predictions = f"pipelines/precipitation_model/impa/predictions_{args.dataset}/{model_name}.hdf"
    if (
        "plot" in specs_dict["models"][model_name].keys()
        and specs_dict["models"][model_name]["plot"] == False
    ):
        continue
    try:
        pred_hdf = h5py.File(predictions)
    except FileNotFoundError:
        print_warning(f"File {predictions} not found. Skipping model {model_name}...")
        continue
    preds.append(pred_hdf)
    model_names.append(model_name)
    log(f"preds: {preds}, \nmodel_names: {model_names}")
# preds = preds[:3]  # TODO: tirar
# model_names = model_names[:3]


# flake8: noqa: C901
def task_lag_img(lag: int):
    future_dt = last_obs_dt + datetime.timedelta(minutes=lag * timestep)
    output_filepath = pathlib.Path(
        f"pipelines/precipitation_model/impa/eval/viz/test/plot-real_time-{args.dataset}/lag={lag}"
    )

    future_time = (future_dt - datetime.timedelta(hours=3)).strftime("%H:%M")
    future_imgs = len(preds) * [np.zeros((HEIGHT, WIDTH, 4), dtype=int)]

    # Predict future
    for i, model_name in enumerate(model_names):
        pred = preds[i]
        if i == 0:
            future_key = last_obs
        else:
            future_key = future_dt.strftime("%Y%m%d-%H%M")
            future_key = f"{last_obs}/{future_key}"
        try:
            if i == 0:
                values = np.array(pred[future_key]).reshape((*latlons.shape[:2], -1))[:, :, 0]
                future_imgs[i] = get_img(
                    values,
                    latlons,
                    "Last observation",
                    feature,
                    None,
                    bg_color=BG_COLOR,
                    height=HEIGHT,
                    width=WIDTH,
                    no_colorbar=True,
                    background=True,
                )
            else:
                values = np.array(pred[future_key])
                future_imgs[i] = get_img(
                    values,
                    latlons,
                    model_name,
                    feature,
                    None,
                    bg_color=BG_COLOR,
                    height=HEIGHT,
                    width=WIDTH,
                    no_colorbar=True,
                    background=True,
                )
        except KeyError:
            pass

    pathlib.Path(output_filepath).parents[0].mkdir(parents=True, exist_ok=True)
    imgs = future_imgs
    for j in range(len(preds)):
        log(f"Start creating image for {model_names[j]}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(imgs[j])
        if j == 0:
            ax.set_facecolor((0.1, 0.2, 0.5))
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        else:
            ax.axis("off")
        ax.axis("tight")

        fig.tight_layout()

        fig.subplots_adjust(bottom=0.1, top=0.9)
        filename = f"{output_filepath}_{model_names[j]}.png"
        log(f"Figure saved on {filename}")
        fig.savefig(filename, transparent=True)
        fig.clf()
        plt.close()


with Pool(min(NLAGS, args.num_workers)) as pool:
    list(tqdm.tqdm(pool.imap(task_lag_img, list(range(NLAGS, NLAGS + 1))), total=NLAGS))
