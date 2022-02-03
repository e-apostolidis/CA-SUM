# -*- coding: utf-8 -*-
import numpy as np
import csv
from scipy.stats import spearmanr, kendalltau, rankdata
from collections import Counter


def evaluate_summary(predicted_summary, user_summary, eval_method):
    """ Compare the predicted summary with the user defined one(s).

    :param np.ndarray predicted_summary: The generated summary from our model.
    :param np.ndarray user_summary: The user defined ground truth summaries (or summary).
    :param str eval_method: The proposed evaluation method; either 'max' (SumMe) or 'avg' (TVSum).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        # Compute precision, recall, f-score
        precision = sum(overlapped)/sum(S)
        recall = sum(overlapped)/sum(G)
        if precision+recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)


def get_corr_coeff(pred_imp_scores, video, dataset):
    """ Read users annotations (frame-level importance scores) for the `video` of the `dataset`* in use. Compare the
    multiple user annotations for the test video with the predicted frame-level importance scores of our CA-SUM for the
    same video, by computing the Spearman's rho and Kendall's tau correlation coefficients. It must be noted, that the
    calculated values are the average correlation coefficients over the multiple annotators.
    * Applicable only for the TVSum dataset.

    :param list[float] pred_imp_scores: The predicted frame-level importance scores from our CA-SUM model.
    :param str video: The name of the test video being inferenced.
    :param str dataset: The dataset in use.
    :return: A tuple containing the video-level Spearman's rho and Kendall's tau correlation coefficients.
    """

    # Read the user annotations from the file
    annot_path = f".../CA-SUM/data/{dataset}/ydata-anno.tsv"
    with open(annot_path) as annot_file:
        user = int(video.split("_")[-1])

        annot = list(csv.reader(annot_file, delimiter="\t"))
        annotation_length = list(Counter(np.array(annot)[:, 0]).values())[user-1]
        init = (user - 1) * annotation_length
        till = user * annotation_length

        user_scores = []
        for row in annot[init:till]:
            curr_user_score = row[2].split(",")
            curr_user_score = np.array([float(num) for num in curr_user_score])
            curr_user_score = curr_user_score / curr_user_score.max(initial=-1)  # Normalize scores between 0 and 1
            curr_user_score = curr_user_score[::15]

            user_scores.append(curr_user_score)

    pred_imp_scores = np.array(pred_imp_scores)
    rho_coeff, tau_coeff = [], []
    for annot in range(len(user_scores)):
        true_user_score = user_scores[annot]
        curr_rho_coeff, _ = spearmanr(pred_imp_scores, true_user_score)
        curr_tau_coeff, _ = kendalltau(rankdata(pred_imp_scores), rankdata(true_user_score))
        rho_coeff.append(curr_rho_coeff)
        tau_coeff.append(curr_tau_coeff)

    rho_coeff = np.array(rho_coeff).mean()  # mean over all user annotations
    tau_coeff = np.array(tau_coeff).mean()  # mean over all user annotations
    return rho_coeff, tau_coeff
