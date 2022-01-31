# -*- coding: utf-8 -*-
# Bash script to automate the procedure of evaluating an experiment.
# First, evaluate each regularization factor of the given experiment. Finally, based only on the training loss and
# through a transductive inference process, choose the best model for this experiment

exp_num=$1
dataset=$2
eval_method=$3  # SumMe -> max | TVSum avg

LC_NUMERIC="en_US.UTF-8"
for sigma in $(seq 0.5 0.1 0.9); do
  sh evaluation/evaluate_factor.sh "$exp_num" "$dataset" "$eval_method" "$sigma"
done

# Run the script that chooses the best model and print the associated metrics
python evaluation/choose_best_model.py "$exp_num" "$dataset"
