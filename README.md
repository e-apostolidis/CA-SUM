<h1><div align="justify"> CA-SUM: Summarizing &nbsp;videos &nbsp;using &nbsp;concentrated attention and considering the uniqueness and diversity of the video frames </div></h1>

## PyTorch Implementation of CA-SUM 
<div align="justify">

- From **"CA-SUM: Summarizing videos using concentrated attention and considering the uniqueness and diversity of the video frames"**.
- Written by Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris and Ioannis Patras.
</div>

## Main dependencies
Developed, checked and verified on an `Ubuntu 20.04.3` PC with an `NVIDIA RTX 2080Ti` GPU and an `i5-11600K` CPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `TensorBoard` | `TensorFlow` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3.8(.8) | 1.7.1 | 11.0 | 8005 | 2.4.0 | 2.4.1 | 1.20.2 | 2.10.0

## Data
<div align="justify">

Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the [data](data) folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
```Text
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth importance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of sub-sampled frames in original video
    /n_steps                  number of sub-sampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
```
Original videos and annotations for each dataset are also available in the dataset providers' webpages: 
- <a href="https://github.com/yalesong/tvsum"><img src="https://img.shields.io/badge/Dataset-TVSum-green"/></a> <a href="https://gyglim.github.io/me/vsum/index.html#benchmark"><img src="https://img.shields.io/badge/Dataset-SumMe-blue"/></a>
</div>

## Configurations
<div align="justify">

Setup for the training process:
 - In [`data_loader.py`](model/data_loader.py), specify the [path to the h5 file of the used dataset](model/data_loader.py#L19:L20), and the [path to the JSON file containing data about the utilized data splits](model/data_loader.py#L21).
 - In [`configs.py`](model/configs.py), define the [directory](model/configs.py#L7) where the analysis results will be saved to. </div>
   
Arguments in [`configs.py`](model/configs.py): 
|Parameter name | Description | Default Value | Options
| :--- | :--- | :---: | :---:
`--mode` | Mode for the configuration. | 'train' | 'train', 'test'
`--verbose` | Print or not training messages. | 'false' | 'true', 'false'
`--video_type` | Used dataset for training the model. | 'SumMe' | 'SumMe', 'TVSum'
`--input_size` | Size of the input feature vectors. | 1024 | int > 0
`--block_size` | Size of the blocks utilized inside the attention matrix. | 60 | 0 < int ≤ 60
`--init_type` | Weight initialization method. | 'xavier' | None, 'xavier', 'normal', 'kaiming', 'orthogonal'
`--init_gain` | Scaling factor for the initialization methods. | √2 | None, float
`--n_epochs` | Number of training epochs. | 400 | int > 0
`--batch_size` | Size of the training batch, 20 for 'SumMe' and 40 for 'TVSum'. | 20 | 0 < int ≤ len(Dataset)
`--seed` | Chosen number for generating reproducible random numbers. | 12345 | None, int
`--clip` | Gradient norm clipping parameter. | 5 | float 
`--lr` | Value of the adopted learning rate. | 5e-4 | float
`--l2_req` | Value of the weight regularization factor. | 1e-5 | float
`--reg_factor` | Value of the length regularization factor. | 0.6 | 0 < float ≤ 1
`--split_index` | Index of the utilized data split. | 0 | 0 ≤ int ≤ 4


## Training
<div align="justify">

To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset (where in each split 80% of the data is used for training and 20% for testing) use the corresponding JSON file that is included in the [data/splits](/data/splits) directory. This file contains the 5 randomly-generated splits that were utilized in our experiments.

For training the model using a single split, run:
```bash
for sigma in $(seq 0.5 0.1 0.9); do
    python model/main.py --split_index N --n_epochs E --batch_size B --video_type 'dataset_name' --reg_factor '$sigma'
done
```
where, `N` refers to the index of the used data split, `E` refers to the number of training epochs, `B` refers to the batch size, `dataset_name` refers to the name of the used dataset, and `'$sigma'` refers to the valid values for the regularization factor hyperparameter.

Alternatively, to train the model for all 5 splits, use the [`run_summe_splits.sh`](model/run_summe_splits.sh) and/or [`run_tvsum_splits.sh`](model/run_tvsum_splits.sh) script and do the following:
```shell-script
chmod +x model/run_summe_splits.sh    # Makes the script executable.
chmod +x model/run_tvsum_splits.sh    # Makes the script executable.
./model/run_summe_splits.sh           # Runs the script. 
./model/run_tvsum_splits.sh           # Runs the script.  
```
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the importance scores for the frames of each video of the test set. These scores are then used by the provided [evaluation](evaluation) scripts to assess the overall performance of the model.

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a command line (cmd) and running: `tensorboard --logdir=/path/to/log-directory --host=localhost`
- opening a browser and pasting the returned URL from cmd. </div>

## Model Selection and Evaluation 
<div align="justify">

The utilized model selection criterion relies on the post-processing of the calculated loss over the training epochs and enables the selection of a well-trained model by indicating the training epoch and the value of the length regularization factor. To evaluate the trained models of the architecture and automatically select a well-trained model, define:
 - the [`dataset_path`](evaluation/compute_fscores.py#L25) in [`compute_fscores.py`](evaluation/compute_fscores.py),
 - the [`base_path`](evaluation/evaluate_factor.sh#L7) in [`evaluate_factor`](evaluation/evaluate_factor.sh),
 - the [`base_path`](evaluation/choose_best_model.py#L12) and [`annot_path`](evaluation/choose_best_model.py#L34) in [`choose_best_model`](evaluation/choose_best_model.py)

and run [`evaluate_exp.sh`](evaluation/evaluate_exp.sh) via
```bash
sh evaluation/evaluate_exp.sh '$exp_num' '$dataset' '$eval_method'
```
where, `'$exp_num'` is the number of the current evaluated experiment, `'$dataset'` refers to the dataset being used, and `'$eval_method'` describe the used approach for computing the overall F-Score after comparing the generated summary with all the available user summaries (i.e., 'max' for SumMe and 'avg' for TVSum).

For further details about the adopted structure of directories in our implementation, please check line [#7](evaluation/evaluate_factor.sh#L7) and line [#13](evaluation/evaluate_factor.sh#L13) of [`evaluate_factor.sh`](evaluation/evaluate_factor.sh). </div>

## License
<div align="justify">

Copyright (c) 2022, Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>
