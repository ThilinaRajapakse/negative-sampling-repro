import os

import datasets
from tqdm.auto import tqdm
import pandas as pd
from huggingface_hub import hf_hub_download


download_dir = "../../data/miracl"
os.makedirs(download_dir, exist_ok=True)

languages = [
    "hi",
    "es",
    "fa",
    "fr",
    "zh",
    "de",
    "yo",
]

print("=== Downloading MIRACL ===")

for lang in tqdm(languages, desc="Downloading corpus"):
    print(f"Downloading corpus for language: {lang}")
    miracl_corpus = datasets.load_dataset("miracl/miracl-corpus", lang)["train"]

    # Rename docid to _id
    miracl_corpus = miracl_corpus.rename_column("docid", "_id")

    # Save to jsonl
    os.makedirs(f"{download_dir}/{lang}", exist_ok=True)
    miracl_corpus.to_json(
        f"{download_dir}/{lang}/corpus.jsonl", orient="records", lines=True
    )


languages = [
    "hi",
    "es",
    "fa",
    "fr",
    "zh",
    "de",
    "yo",
]

for lang in languages:
    print(f"Downloading qrels for language: {lang}")
    qrels_path = hf_hub_download(
        repo_id="miracl/miracl",
        filename=f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv",
        repo_type="dataset",
    )

    qrels = pd.read_csv(qrels_path, sep="\t", header=None)
    qrels.columns = ["query-id", "NA", "corpus-id", "score"]
    qrels = qrels[["query-id", "corpus-id", "score"]]
    os.makedirs(f"{download_dir}/{lang}/qrels", exist_ok=True)
    qrels.to_csv(
        f"{download_dir}/{lang}/qrels/dev.tsv", sep="\t", header=False, index=False
    )

qrels

languages = [
    "de",
    "es",
    "fa",
    "fr",
    "hi",
    "yo",
    "zh",
]

for lang in languages:
    print(f"Downloading topics for language: {lang}")
    qrels_path = hf_hub_download(
        repo_id="miracl/miracl",
        filename=f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv",
        repo_type="dataset",
    )

    topics = pd.read_csv(qrels_path, sep="\t", header=None)
    topics.columns = ["_id", "text"]
    topics.to_json(f"{download_dir}/{lang}/queries.jsonl", orient="records", lines=True)


print("=== MIRACL download complete ===")
