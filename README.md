# Negative Sampling Techniques for Dense Passage Retrieval in a Multilingual Setting
## A Reproducibility Study


This repository contains the code and data for the paper "Negative Sampling Techniques for Dense Passage Retrieval in a Multilingual Setting" submitted to the Reproducibility Track of SIGIR 2024.

## Setup

To install the required packages, run the following commands:

```bash
conda create -n ns python=3.10 pandas tqdm
conda activate ns
conda install pytorch=2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl

pip install transformers
pip install datasets
pip install simpletransformers
```

## Data

Run download_data.sh to download the data.

```bash
bash download_data.sh
```

## Training

To train the models, run the following commands:

```bash
bash train.sh
```

You can edit the train.sh file to select the models you want to train.

The hyperparameters can be changed in the training scripts. The hyperparameters are set to the values used in the paper by default.

## Evaluation

To evaluate the models, run the following commands:

```bash
bash evaluate.sh
```

You can edit the evaluate.sh file to select the models you want to evaluate.

## Results

The results will be saved in the `results` directory.
