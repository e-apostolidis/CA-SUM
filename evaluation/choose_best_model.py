# -*- coding: utf-8 -*-
import numpy as np
import csv
import json
import sys
from scipy.stats import spearmanr, kendalltau, rankdata

# example usage: python evaluation/choose_best_model.py 500 "TVSum"
exp_num = sys.argv[1]
dataset = sys.argv[2]

base_path = ".../CA-SUM/Summaries"
eligible_datasets = ["TVSum"]


def get_corr_coeff(epoch, split_id, reg_factor):
    """ Read users annotations (frame-level importance scores) for each video in the dataset*. Compare the multiple
    user annotations for each test video with the predicted frame-level importance scores of our CA-SUM for the same
    video, by computing the Spearman's rho and Kendall's tau correlation coefficients. It must be noted, that for each
    test video the calculated values are the average correlation coefficients over the multiple annotators. The final
    split-level values are the average over the entire test set.
    * Applicable only for the TVSum dataset.

    :param int epoch: The chosen training epoch for the given split and regularization factor.
    :param int split_id: The id of the current evaluated split.
    :param float reg_factor: The value of the current evaluated length regularization factor.
    :return: A tuple containing the split-level Spearman's rho and Kendall's tau correlation coefficients.
    """
    if dataset not in eligible_datasets:
        print(f"Correlation coefficients are not supported by {dataset} dataset.")
        return None, None

    # Read the user annotations from the file
    annot_path = f".../CA-SUM/data/{dataset}/annotations/ydata-anno.tsv"
    with open(annot_path) as annot_file:
        annot = csv.reader(annot_file, delimiter="\t")
        names, user_scores = [], {}
        for row in annot:
            str_user = row[0]

            curr_user_score = row[2].split(",")
            curr_user_score = np.array([float(num) for num in curr_user_score])
            curr_user_score = curr_user_score / curr_user_score.max(initial=-1)  # Normalize scores between 0 and 1
            curr_user_score = curr_user_score[::15]

            if str_user not in names:
                names.append(str_user)
                user = f"video_{len(names)}"
                user_scores[user] = [curr_user_score]
            else:
                user_scores[user].append(curr_user_score)

    # Read each score and compared it
    scores_path = f"{base_path}/exp{exp_num}/reg{reg_factor}/{dataset}/results/split{split_id}/{dataset}_{epoch-1}.json"
    with open(scores_path) as score_file:       # Read the importance scores affiliated with the selected epoch
        scores = json.loads(score_file.read())
        keys = list(scores.keys())

    rho_coeff_video, tau_coeff_video = [], []
    for video in keys:
        pred_imp_score = np.array(scores[video])
        curr_user_scores = user_scores[video]

        rho_coeff, tau_coeff = [], []
        for annot in range(len(curr_user_scores)):
            true_user_score = curr_user_scores[annot]
            curr_rho_coeff, _ = spearmanr(pred_imp_score, true_user_score)
            curr_tau_coeff, _ = kendalltau(rankdata(pred_imp_score), rankdata(true_user_score))
            rho_coeff.append(curr_rho_coeff)
            tau_coeff.append(curr_tau_coeff)

        rho_coeff = np.array(rho_coeff).mean()  # mean over all user annotations
        rho_coeff_video.append(rho_coeff)
        tau_coeff = np.array(tau_coeff).mean()  # mean over all user annotations
        tau_coeff_video.append(tau_coeff)

    rho_coeff_split = np.array(rho_coeff_video).mean()  # mean over all videos
    tau_coeff_split = np.array(tau_coeff_video).mean()  # mean over all videos

    return rho_coeff_split, tau_coeff_split


def get_improvement_score(epoch, split_id, reg_factor):
    """ Using the estimated frame-level importance scores from an untrained model, calculate the improvement (eq. 2-3)
    of a  trained model for the chosen epoch, on a given split and regularization factor.

    :param int epoch: The chosen training epoch for the given split and regularization factor.
    :param int split_id: The id of the current evaluated split.
    :param float reg_factor: The value of the current evaluated length regularization factor
    :return: The relative improvement of a trained model over an untrained (random) one.
    """
    untr_path = f"{base_path}/exp{exp_num}/reg{reg_factor}/{dataset}/results/split{split_id}/{dataset}_-1.json"
    curr_path = f"{base_path}/exp{exp_num}/reg{reg_factor}/{dataset}/results/split{split_id}/{dataset}_{epoch}.json"
    with open(curr_path) as curr_file, open(untr_path) as untr_file:
        untr_data = json.loads(untr_file.read())
        curr_data = json.loads(curr_file.read())

        keys = list(curr_data.keys())
        mean_untr_scores, mean_curr_scores = [], []
        for video_name in keys:                              # For a video inside that split get the ...
            untr_scores = np.asarray(untr_data[video_name])  # Untrained model computed importance scores
            curr_scores = np.asarray(curr_data[video_name])  # trained model computed importance scores

            mean_untr_scores.append(np.mean(untr_scores))
            mean_curr_scores.append(np.mean(curr_scores))

    mean_untr_scores = np.array(mean_untr_scores)
    mean_curr_scores = np.array(mean_curr_scores)

    # Measure how much did we improve a random model, relatively to moving towards sigma (minimum loss)
    improvement = abs(mean_curr_scores.mean() - mean_untr_scores.mean())
    result = (improvement / abs(reg_factor - mean_untr_scores.mean()))
    return result


def train_logs(log_file, method="argmin"):
    """ Choose and return the epoch based only on the training loss. Through the `method` argument you can get the epoch
    associated with the minimum training loss (argmin) or the last epoch of the training process (last).

    :param str log_file: Path to the saved csv file containing the loss information.
    :param str method: The chosen criterion for the epoch (model) picking process.
    :return: The epoch of the best model, according to the chosen criterion.
    """
    losses = {}
    losses_names = []

    # Read the csv file with the training losses
    with open(log_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for (i, row) in enumerate(csv_reader):
            if i == 0:
                for col in range(len(row)):
                    losses[row[col]] = []
                    losses_names.append(row[col])
            else:
                for col in range(len(row)):
                    losses[losses_names[col]].append(float(row[col]))

    # criterion: The length regularization of the generated summary (400 epochs, after which overfitting problems occur)
    loss = losses["loss_epoch"]
    loss = loss[:400]
    START_EPOCH = 20                      # If unstable training is observed at the start

    if method == "last":
        epoch = len(loss) - 1
    elif method == "argmin":
        epoch = np.array(loss[START_EPOCH:]).argmin() + START_EPOCH
    else:
        raise ValueError(f"Method {method} is not currently supported. Only `last` and `argmin` are available.")

    return epoch


# Choose the model associated with the min training loss for each regularization factor and get its improvement score
all_improvements, all_epochs = [], []
sigmas = [i/10 for i in range(5, 10)]  # The valid values for the length regularization factor
for sigma in sigmas:
    split_improvements, split_epochs = np.zeros(5, dtype=float), np.zeros(5, dtype=int)
    for split in range(0, 5):
        log = f"{base_path}/exp{exp_num}/reg{sigma}/{dataset}/logs/split{split}/scalars.csv"
        selected_epoch = train_logs(log, method="argmin")  # w/o +1. (only needed to pick the f-score value)

        split_improvements[split] = get_improvement_score(epoch=selected_epoch, split_id=split, reg_factor=sigma)
        split_epochs[split] = selected_epoch
    all_improvements.append(split_improvements)
    all_epochs.append(split_epochs)

# From list to nd array for easier computations
all_improvements = np.stack(all_improvements)
all_epochs = np.stack(all_epochs)

# Choose the highest improvement sigma's per split
all_improvements = np.where(all_improvements > 1.5, 0, all_improvements)
print(all_improvements)
improvement_per_spit = all_improvements.max(axis=0, initial=-1)
chosen_indices = all_improvements.argmax(axis=0)
sigma_per_split = np.array(sigmas)[chosen_indices]

# For the chosen epochs and length regularization factors, calculate the metrics for our assessments
all_fscores, all_rho_coeff, all_tau_coeff = np.zeros(5, dtype=float), np.zeros(5, dtype=float), np.zeros(5, dtype=float)
for split in range(0, 5):
    curr_sigma = sigma_per_split[split]
    curr_epoch = all_epochs[chosen_indices[split], split] + 1  # because of the evaluation on the untrained model

    # Read the fscore values
    results_file = f"{base_path}/exp{exp_num}/reg{curr_sigma}/{dataset}/results/split{split}/f_scores.txt"
    with open(results_file) as f:
        f_scores = f.read().strip()  # read F-Scores
        if "\n" in f_scores:
            f_scores = f_scores.splitlines()
        else:
            f_scores = json.loads(f_scores)

    f_scores = np.array([float(f_score) for f_score in f_scores])
    curr_fscore = np.round(f_scores[curr_epoch], 2)
    all_fscores[split] = curr_fscore
    print(f"[Split: {split}] Fscore: {curr_fscore:.2f}", end="")

    # Compute correlation coefficients
    if dataset in eligible_datasets:
        rho, tau = get_corr_coeff(epoch=curr_epoch, split_id=split, reg_factor=curr_sigma)
        all_rho_coeff[split] = rho
        all_tau_coeff[split] = tau
        print(f"  Spearman's \u03C1: {rho:.3f}  Kendall's \u03C4: {tau:.3f}", end="")
    print(f" [\u03C3={curr_sigma}, epoch: {curr_epoch}]")

avg_fscore = np.round(np.mean(all_fscores), 2)
if dataset in eligible_datasets:
    avg_rho, avg_tau = np.round(np.mean(all_rho_coeff), 2), np.round(np.mean(all_tau_coeff), 2)
    print("====================================================================================")
    print(f"Avg values :=> F1: {avg_fscore}  Spearman's \u03C1: {avg_rho:.3f}  Kendall's \u03C4: {avg_tau:.3f}")
else:
    print(f"Avg values :=> F1: {avg_fscore:.2f}")
