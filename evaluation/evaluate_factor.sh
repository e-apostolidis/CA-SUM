# -*- coding: utf-8 -*-
# The script is taking as granted that the experiments are saved under "base_path" as "base_path/exp$EXP_NUM".
# Run the evaluation script of the `exp_num` experiment for a specific `dataset` and a `regularization factor`.
# First, get the training loss from tensorboard as a csv file, for each data-split for the given regularization factor.
# Then, compute the fscore (txt file) associated with the above mentioned data-splits and regularization factor.

base_path=".../CA-SUM/Summaries/"
exp_num=$1
dataset=$2
eval_method=$3  # SumMe -> max | TVSum avg
factor=$4

exp_path="$base_path/exp$exp_num/reg$factor"; echo "$exp_path"  # add factor to the path of the experiment

for i in 0 1 2 3 4; do
  path="$exp_path/$dataset/logs/split$i"
  python evaluation/exportTensorFlowLog.py "$path" "$path"
  results_path="$exp_path/$dataset/results/split$i"
  python evaluation/compute_fscores.py --path "$results_path" --dataset "$dataset" --eval "$eval_method"
done
