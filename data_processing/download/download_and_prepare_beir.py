import os

import datasets
import pandas as pd
from tqdm.auto import tqdm


os.makedirs("../../data/beir", exist_ok=True)

print("=== Downloading BEIR ===")

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


for dataset in tqdm(all_datasets):
    print(f"Downloading dataset: {dataset}")
    os.makedirs(f"../../data/beir/{dataset}", exist_ok=True)

    queries = datasets.load_dataset(f"BeIR/{dataset}", "queries")["queries"]
    corpus = datasets.load_dataset(f"BeIR/{dataset}", "corpus")["corpus"]
    qrels = datasets.load_dataset(f"BeIR/{dataset}-qrels", "qrels")["test"]

    queries.to_json(
        f"../../data/beir/{dataset}/queries.jsonl", orient="records", lines=True
    )
    corpus.to_json(
        f"../../data/beir/{dataset}/corpus.jsonl", orient="records", lines=True
    )

    os.makedirs(f"../../data/beir/{dataset}/qrels", exist_ok=True)
    qrels.to_csv(f"../../data/beir/{dataset}/qrels/test.tsv", sep="\t", index=False)

print("=== BEIR download complete ===")
