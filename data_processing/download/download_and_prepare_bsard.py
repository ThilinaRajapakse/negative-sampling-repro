import os
import datasets
import pandas as pd


def download_and_prepare_bsard():
    """
    Downloads and prepares the BSARD dataset from the Hugging Face library.
    The dataset is saved as JSONL files for the corpus and queries, and a TSV file for the qrels.
    """

    print("=== Downloading BSARD ===")

    # Load the BSARD corpus dataset
    dataset = datasets.load_dataset("maastrichtlawtech/bsard", "corpus")

    # Create the output directory if it doesn't exist
    os.makedirs("../../data/bsard", exist_ok=True)

    # Prepare the corpus dataframe
    corpus_df = pd.DataFrame(dataset["corpus"])
    corpus_df = corpus_df[["id", "article"]]
    corpus_df.columns = ["_id", "text"]

    # Save the corpus dataframe as a JSONL file
    corpus_df.to_json("../../data/bsard/corpus.jsonl", orient="records", lines=True)

    # Load the BSARD questions dataset
    dataset = datasets.load_dataset("maastrichtlawtech/bsard", "questions")
    dataset = dataset["test"]

    # Prepare the queries dataframe
    queries_df = pd.DataFrame(dataset)
    queries_df = queries_df[["id", "question"]]
    queries_df.columns = ["_id", "text"]

    # Save the queries dataframe as a JSONL file
    queries_df.to_json("../../data/bsard/queries.jsonl", orient="records", lines=True)

    # Prepare the qrels dataframe
    qrel_dict = {"query-id": [], "corpus-id": [], "score": []}

    # Generate the qrels dataframe
    for row in dataset:
        for article in row["article_ids"]:
            qrel_dict["query-id"].append(row["id"])
            qrel_dict["corpus-id"].append(article)
            qrel_dict["score"].append(1)

    qrel_df = pd.DataFrame(qrel_dict)

    # Create the qrels directory if it doesn't exist
    os.makedirs("../../data/bsard/qrels", exist_ok=True)

    # Save the qrels dataframe as a TSV file
    qrel_df.to_csv("../../data/bsard/qrels/test.tsv", index=False, sep="\t")

    print("=== BSARD download complete ===")


# Call the function to download and prepare the BSARD dataset
download_and_prepare_bsard()
