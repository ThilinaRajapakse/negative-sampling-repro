import os
import sys
import json
import logging
from functools import partial

import torch
import numpy as np
from tqdm.auto import tqdm

from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
from simpletransformers.retrieval.retrieval_utils import get_output_embeddings, embed
from datasets import Dataset as HFDataset


logging.basicConfig(level=logging.INFO)


dataset_collection = sys.argv[1]

if dataset_collection == "beir":
    all_datasets = [
        "scifact",
        "arguana",
        "scidocs",
        "fiqa",
        "nf_corpus",
        "quora",
        "dbpedia",
        "hotpotqa",
        "trec_covid",
        "cqa_dup_stack",
    ]
elif dataset_collection == "tydi":
    all_datasets = [
        "swahili",
        "telugu",
        "indonesian",
        "bengali",
        "korean",
        "arabic",
        "finnish",
        "japanese",
        "russian",
        "thai",
    ]
elif dataset_collection == "mmarco":
    all_datasets = [
        "chinese",
        "arabic",
        "indonesian",
        "korean",
        "japanese",
        "russian",
        "french",
        "dutch",
        "german",
        "italian",
        "portuguese",
        "spanish",
        "hindi",
        "vietnamese",
    ]

    ood_langs = [
        "arabic",
        "english",
        "indonesian",
        "japanse",
        "russian",
    ]
elif dataset_collection == "miracl":
    all_datasets = [
        "hi",
        "es",
        "fa",
        "fr",
        "zh",
        "de",
        "yo",
    ]
elif dataset_collection == "other":
    all_datasets = [
        "bsard",
        "gerdalir",
        "multi_cpr_video",
        "multi_cpr_ecom",
    ]


dataset_no = int(sys.argv[2])
dataset = all_datasets[dataset_no]

if dataset_collection == "other":
    data_path = f"../../data/{dataset}"
    results_dir = f"../results/other-zero/"
else:
    data_path = f"../../data/{dataset_collection}/{dataset}"
    if dataset_collection == "mmarco":
        if dataset in ood_langs:
            results_dir = f"../results/{dataset_collection}-out-of-distribution"
        else:
            results_dir = f"../results/{dataset_collection}-in-distribution"
    elif dataset_collection == "miracl":
        results_dir = f"../results/{dataset_collection}"
    elif dataset_collection == "beir":
        results_dir = f"../results/{dataset_collection}"
    elif dataset_collection == "tydi":
        results_dir = f"../results/{dataset_collection}-in-domain"


all_models = [
    "../../trained_models/finetuned/ANCE-tydi",
    "../../trained_models/finetuned/DPR-base-tydi",
    "../../trained_models/finetuned/DPR-BM-tydi",
    "../../trained_models/finetuned/ICT-passage-tydi",
    "../../trained_models/finetuned/ICT-query-tydi",
    "../../trained_models/finetuned/TAS-passage-tydi",
    "../../trained_models/finetuned/TAS-query-tydi",
]

model_no = int(sys.argv[3])
model_path = all_models[model_no]


model_type = "custom"
context_name = f"{model_path}/context_encoder"
question_name = f"{model_path}/query_encoder"
model_name = None

model_args = RetrievalArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_cached_eval_features = False

model_args.retrieve_n_docs = 100
model_args.max_seq_length = 256
model_args.eval_batch_size = 100
model_args.use_hf_datasets = True
model_args.include_title = False
model_args.include_title_in_corpus = False
model_args.n_gpu = 1
model_args.data_format = "beir"

model_args.output_dir = f"../indices/mmarco/{dataset}/{os.path.split(model_path)[-1]}"


model = RetrievalModel(
    model_type=model_type,
    model_name=model_name,
    context_encoder_name=context_name,
    query_encoder_name=question_name,
    args=model_args,
)

report = model.eval_model(
    data_path,
    save_as_experiment=True,
    experiment_name=results_dir,
    dataset_name=dataset,
    model_name=os.path.split(model_path)[-1],
    eval_set="test",
    pytrec_eval_metrics=["recip_rank", "recall_100", "ndcg_cut_10"],
)

print("#############################")
print(f"Dataset: {dataset}")
print(report)
