#!/bin/bash

# Define all datasets
datasets=("tydi", "mmarco", "miracl", "other", "beir")

# Define all models
models=("ANCE-tydi" "DPR-base-tydi" "DPR-BM-tydi" "ICT-passage-tydi" "ICT-query-tydi" "TAS-passage-tydi" "TAS-query-tydi")

# Iterate over all combinations of datasets and models
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # Call your Python script here
        python eval_all.py "$dataset" "$model"
    done
done