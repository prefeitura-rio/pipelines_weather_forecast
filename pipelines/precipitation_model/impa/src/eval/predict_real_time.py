# -*- coding: utf-8 -*-
# flake8: noqa: E501

import json
import pathlib
from argparse import ArgumentParser

from src.utils.eval_utils import predict_dict
from src.utils.general_utils import print_warning


def predict(num_workers=8, cuda=False):
    accelerator = "gpu" if cuda else "cpu"

    config = pathlib.Path("src/eval/real_time_config.json")
    with open(config, "r") as json_file:
        specs_dict = json.load(json_file)

    for model_name, info in specs_dict["models"].items():
        if "needs_cuda" in info and info["needs_cuda"] and not cuda:
            print_warning(f"Skipping {model_name} because it needs CUDA...")
            continue

        predict_func = predict_dict[model_name]
        if "args" in info:
            args = info["args"]
            args["num_workers"] = num_workers
            args["accelerator"] = accelerator

            if "params_filepath" in info:
                with open(info["params_filepath"], "r") as json_file:
                    params = json.load(json_file)
                args |= params
                if model_name in ["UNET", "EVONET", "NowcastNet", "MetNet3", "MetNet_lead_time"]:
                    args["model"] = args["model_name"]
                del params["model_name"]
                predict_func(args, params)
            else:
                predict_func(args)
            continue

        model_path = pathlib.Path(f"models/{model_name}/")
        model_file = model_path / info["model_file"]

        output_predict_filepaths = [f"predictions/{model_name}.hdf"]

        # Standard arguments
        args = {
            "overwrite": True,
            "locations": ["rio_de_janeiro"],
            "accelerator": accelerator,
            "output_predict_filepaths": output_predict_filepaths,
            "dataframe_filepath": "data/dataframes/SAT-CORRECTED-ABI-L2-RRQPEF-real_time-{location}/test.hdf",
            "dataframe": "SAT-ABI-L2-RRQPEF-{location}-file=thr=0_split2",
            "num_workers": num_workers,
            "input_model_filepath": model_file,
            "compile": False,
            "batch_to_predict": 8,
        }
        # Model params
        with open(model_path / "params.json", "r") as json_file:
            params = json.load(json_file)
        args |= params
        ##################################
        if model_name in ["UNET", "EVONET", "NowcastNet", "MetNet3", "MetNet_lead_time"]:
            args["model"] = args["model_name"]
        ##################################
        del args["model_name"]
        predict_func(args, params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    predict(**vars(args))
