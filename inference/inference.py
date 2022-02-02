# -*- coding: utf-8 -*-
import torch
import numpy as np
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary, get_corr_coeff
from layers.summarizer import CA_SUM
from os import listdir
from os.path import isfile, join
import h5py
import json
import argparse

eligible_datasets = ["TVSum"]


def inference(model, data_path, keys, eval_method):
    """ Used to inference a pretrained `model` on the `keys` test videos, based on the `eval_method` criterion; using
        the dataset located in `data_path'.

        :param nn.Module model: Pretrained model to be inferenced.
        :param str data_path: File path for the dataset in use.
        :param list keys: Containing the test video keys of the used data split.
        :param str eval_method: The evaluation method in use {SumMe: max, TVSum: avg}.
    """
    model.eval()
    video_fscores, video_rho, video_tau = [], [], []
    for video in keys:
        with h5py.File(data_path, "r") as hdf:
            # Input features for inference
            frame_features = torch.Tensor(np.array(hdf[f"{video}/features"])).view(-1, 1024)
            frame_features = frame_features.to(model.linear_1.weight.device)

            # Input need for evaluation
            user_summary = np.array(hdf[f"{video}/user_summary"])
            sb = np.array(hdf[f"{video}/change_points"])
            n_frames = np.array(hdf[f"{video}/n_frames"])
            positions = np.array(hdf[f"{video}/picks"])

        with torch.no_grad():
            scores, _ = model(frame_features)  # [1, seq_len]
            scores = scores.squeeze(0).cpu().numpy().tolist()
            summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
            f_score = evaluate_summary(summary, user_summary, eval_method)
            video_fscores.append(f_score)

            if dataset in eligible_datasets:
                rho, tau = get_corr_coeff(pred_imp_scores=scores, video=video, dataset=dataset)
                video_rho.append(rho)
                video_tau.append(tau)

    print(f"CA-SUM model trained for split: {split_id} achieved an F-score: {np.mean(video_fscores):.2f}%", end="")
    if dataset not in eligible_datasets:
        print("\n", end="")
    else:
        print(f", a Spearman's \u03C1: {np.mean(video_rho):.3f}  and a Kendall's \u03C4: {np.mean(video_tau):.3f}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # arguments to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='SumMe', help="Dataset to be used. Supported: {SumMe, TVSum}")

    args = vars(parser.parse_args())
    dataset = args["dataset"]

    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'
    for split_id in range(5):
        # Model data
        model_path = f".../CA-SUM/inference/pretrained_models/{dataset}/split{split_id}"
        model_file = [f for f in listdir(model_path) if isfile(join(model_path, f))]

        # Read current split
        split_file = f".../CA-SUM/data/splits/{dataset.lower()}_splits.json"
        with open(split_file) as f:
            data = json.loads(f.read())
            test_keys = data[split_id]["test_keys"]

        # Dataset path
        dataset_path = f".../CA-SUM/data/{dataset}/eccv16_dataset_{dataset.lower()}_google_pool5.h5"

        # Create model with paper reported configuration
        trained_model = CA_SUM(input_size=1024, output_size=1024, block_size=60).to(device)
        trained_model.load_state_dict(torch.load(join(model_path, model_file[-1])))
        inference(trained_model, dataset_path, test_keys, eval_metric)
