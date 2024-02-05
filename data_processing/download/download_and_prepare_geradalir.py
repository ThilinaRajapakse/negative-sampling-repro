import os
import pandas as pd


def download_and_prepare_geradalir():
    """
    This function reads the corpus, queries, and qrels files for the Geradalir dataset,
    performs data processing and preparation, and saves the processed data to the appropriate files.
    """

    print("=== Downloading Geradalir ===")
    # Read corpus file
    corpus_df = pd.read_csv(
        "../raw_data/gerdalir/collection.tsv",
        sep="\t",
        header=None,
        names=["_id", "text"],
        on_bad_lines="warn",
        encoding="utf-8",
    )

    # Read queries file
    query_df = pd.read_csv(
        "../raw_data/gerdalir/queries/queries.test.tsv",
        sep="\t",
        header=None,
        names=["_id", "text"],
        on_bad_lines="warn",
        encoding="utf-8",
    )

    # Read qrels file
    test_qrels_df = pd.read_csv(
        "../raw_data/gerdalir/qrels/qrels.test.tsv",
        sep="\t",
        header=None,
        names=["query-id", "corpus-id"],
        on_bad_lines="warn",
        encoding="utf-8",
    )
    test_qrels_df["score"] = 1

    # Check if all corpus-ids in qrels are in corpus
    test_qrels_df["corpus-id"].isin(corpus_df["_id"]).value_counts()

    # Check if all query-ids in qrels are in queries
    test_qrels_df["query-id"].isin(query_df["_id"]).value_counts()

    # Drop all qrels with corpus-ids not in corpus
    test_qrels_df = test_qrels_df[test_qrels_df["corpus-id"].isin(corpus_df["_id"])]

    # Drop all qrels with query-ids not in queries
    test_qrels_df = test_qrels_df[test_qrels_df["query-id"].isin(query_df["_id"])]

    # Check if all corpus-ids in qrels are in corpus
    assert test_qrels_df["corpus-id"].isin(corpus_df["_id"]).value_counts()[
        True
    ] == len(test_qrels_df)

    # Check if all query-ids in qrels are in queries
    assert test_qrels_df["query-id"].isin(query_df["_id"]).value_counts()[True] == len(
        test_qrels_df
    )

    # Drop any rows with missing values
    test_qrels_df.dropna(inplace=True)

    # Create directories if they don't exist
    os.makedirs("../../data/gerdalir", exist_ok=True)
    os.makedirs("../../data/gerdalir/qrels", exist_ok=True)

    # Save processed data to files
    corpus_df.to_json("../../data/gerdalir/corpus.jsonl", orient="records", lines=True)
    query_df.to_json("../../data/gerdalir/queries.jsonl", orient="records", lines=True)
    test_qrels_df.to_csv("../../data/gerdalir/qrels/test.tsv", sep="\t", index=False)

    print("=== Geradalir download complete ===")


# Call the function to execute the code
download_and_prepare_geradalir()
