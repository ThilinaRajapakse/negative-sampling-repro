import os

import datasets
import pandas as pd
from tqdm.auto import tqdm


def download_and_prepare_tydi_train():
    """
    Downloads and prepares the Mr. TyDi training dataset in a specific format.

    This function downloads the Mr. TyDi dataset for multiple languages, extracts the relevant information,
    and saves it in a specific format for training purposes.

    Returns:
        None
    """
    print("=== Downloading Mr. TyDi ===")

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

    split = "train"

    for i, language in enumerate(languages):
        print("---------------------------------------")
        print(f"Language: {language}")
        print(f"Language {i+1} of {len(languages)}")
        query_dataset = datasets.load_dataset("castorini/mr-tydi", language)

        query_dataset

        os.makedirs(f"../../data/tydi/{language}/", exist_ok=True)

        queries_dict = {
            "query_text": [],
            "gold_passage": [],
            "title": [],
            "hard_negative": [],
            "query_id": [],
        }

        for query in tqdm(query_dataset[split]):
            queries_dict["query_id"].append(query["query_id"])
            queries_dict["query_text"].append(query["query"])
            queries_dict["gold_passage"].append(query["positive_passages"][0]["text"])
            queries_dict["title"].append(query["positive_passages"][0]["title"])
            try:
                queries_dict["hard_negative"].append(
                    query["negative_passages"][0]["title"]
                    + " "
                    + query["negative_passages"][0]["text"]
                )
            except:
                queries_dict["hard_negative"].append("")

        query_df = pd.DataFrame(queries_dict)
        query_df.to_csv(
            f"../../data/tydi/{language}/{split}.tsv", sep="\t", index=False
        )

        print("---------------------------------------")

    print("Combining all languages...")

    languages = os.listdir("../../data/tydi")
    languages = [
        language
        for language in languages
        if os.path.isdir(f"../../data/tydi/{language}")
    ]

    train_type = "train"

    tydi_dfs = []
    for language in languages:
        tydi_df = pd.read_csv(f"../../data/tydi/{language}/{train_type}.tsv", sep="\t")
        tydi_df["gold_passage"] = tydi_df["title"] + " " + tydi_df["gold_passage"]
        tydi_df = tydi_df[["query_text", "gold_passage", "hard_negative"]]
        print(f"Language: {language}")
        print(f"Size: {len(tydi_df)}")
        tydi_dfs.append(tydi_df)

    combined_df = pd.concat(tydi_dfs)

    combined_df.to_csv(f"../data/tydi/{train_type}_tydi.tsv", sep="\t", index=False)

    print("Combined TyDi dataset saved to ../data/tydi/tydi-train.tsv")

    print("=== Mr. TyDi download complete ===")


download_and_prepare_tydi_train()
