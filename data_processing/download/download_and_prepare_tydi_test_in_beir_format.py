import json
import os

import datasets
import pandas as pd
from tqdm.auto import tqdm


def download_and_prepare_tydi_test():
    """
    Downloads the Mr. TyDi Test dataset for multiple languages and prepares it in the BEIR format.

    This script downloads the Mr. TyDi Test dataset for multiple languages and prepares it in the BEIR format.
    It generates the qrels (relevance judgments) file, queries file, and corpus file for each language.

    Returns:
        None
    """
    split = "test"

    print("=== Downloading Mr. TyDi Test ===")

    languages = [
        "swahili",
        "bengali",
        "telugu",
        "thai",
        "arabic",
        "finnish",
        "indonesian",
        "japanese",
        "korean",
        "russian",
    ]

    for i, language in enumerate(languages):
        print("=======================================")
        print(f"Language: {language}")
        print(f"Language {i+1} of {len(languages)}")
        query_dataset = datasets.load_dataset("castorini/mr-tydi", language)

        corpus_dataset = datasets.load_dataset("castorini/mr-tydi-corpus", language)
        print("Loaded corpus of size", len(corpus_dataset["train"]))
        print("---------------------------------------")

        # Build qrels tsv
        qrels_dict = {
            "query-id": [],
            "corpus-id": [],
            "score": [],
        }

        for query in tqdm(query_dataset[split]):
            qrels_dict["query-id"].append(query["query_id"])
            qrels_dict["corpus-id"].append(query["positive_passages"][0]["docid"])
            qrels_dict["score"].append(1)

        qrels_df = pd.DataFrame(qrels_dict)
        qrels_df

        os.makedirs(f"../../data/tydi/{language}/qrels", exist_ok=True)
        qrels_df.to_csv(
            f"../../data/tydi/{language}/qrels/{split}.tsv", sep="\t", index=False
        )

        queries_dict = {
            "_id": [],
            "text": [],
        }

        for query in tqdm(query_dataset[split]):
            queries_dict["_id"].append(query["query_id"])
            queries_dict["text"].append(query["query"])

        query_df = pd.DataFrame(queries_dict)
        query_df

        query_df.to_json(
            f"../../data/tydi/{language}/queries.jsonl", orient="records", lines=True
        )

        corpus_dataset = corpus_dataset["train"]
        corpus_dataset = corpus_dataset.rename_column("docid", "_id")

        corpus_dataset.to_json(
            f"../../data/tydi/{language}/corpus.jsonl", orient="records", lines=True
        )

    print("=== Mr. TyDi Test download complete ===")


# Call the function to execute the code
download_and_prepare_tydi_test()
