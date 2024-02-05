import os
import pandas as pd


def process_multi_cpr():
    """
    Process Multi-CPR data by reading input files, converting them to desired formats,
    and saving the processed data to output files.
    """
    print("=== Processing Multi-CPR ===")

    # Read ecom corpus
    ecom_corpus = pd.read_csv(
        "..//Multi-CPR/data/ecom/corpus.tsv",
        sep="\t",
        header=None,
        names=["_id", "text"],
    )

    # Read ecom queries
    queries_df = pd.read_csv(
        "/../Multi-CPR/data/ecom/dev.query.txt",
        sep="\t",
        header=None,
        names=["_id", "text"],
    )

    # Read ecom qrels
    qrels_df = pd.read_csv(
        "/../Multi-CPR/data/ecom/qrels.dev.tsv",
        sep="\t",
        header=None,
        names=["query-id", "NA", "corpus-id", "score"],
    )
    qrels_df = qrels_df.drop(columns=["NA"])

    # Create directories for output files
    os.makedirs("../../data/multi_cpr_ecom", exist_ok=True)

    # Save ecom corpus to JSONL file
    ecom_corpus.to_json(
        "../../data/multi_cpr_ecom/corpus.jsonl", orient="records", lines=True
    )

    # Save ecom queries to JSONL file
    queries_df.to_json(
        "../../data/multi_cpr_ecom/queries.jsonl", orient="records", lines=True
    )

    # Create directories for qrels files
    os.makedirs("../../data/multi_cpr_ecom/qrels", exist_ok=True)

    # Save ecom qrels to TSV file
    qrels_df.to_csv("../../data/multi_cpr_ecom/qrels/dev.tsv", sep="\t", index=False)

    # Read video corpus
    video_corpus = pd.read_csv(
        "../Multi-CPR/data/video/corpus.tsv",
        sep="\t",
        header=None,
        names=["_id", "text"],
    )

    # Read video queries
    queries_df = pd.read_csv(
        "../Multi-CPR/data/video/dev.query.txt",
        sep="\t",
        header=None,
        names=["_id", "text"],
    )

    # Read video qrels
    qrels_df = pd.read_csv(
        "../Multi-CPR/data/video/qrels.dev.tsv",
        sep="\t",
        header=None,
        names=["query-id", "NA", "corpus-id", "score"],
    )
    qrels_df = qrels_df.drop(columns=["NA"])

    # Create directories for output files
    os.makedirs("../../data/multi_cpr_video", exist_ok=True)

    # Save video corpus to JSONL file
    video_corpus.to_json(
        "../../data/multi_cpr_video/corpus.jsonl", orient="records", lines=True
    )

    # Save video queries to JSONL file
    queries_df.to_json(
        "../../data/multi_cpr_video/queries.jsonl", orient="records", lines=True
    )

    # Create directories for qrels files
    os.makedirs("../../data/multi_cpr_video/qrels", exist_ok=True)

    # Save video qrels to TSV file
    qrels_df.to_csv("../../data/multi_cpr_video/qrels/dev.tsv", sep="\t", index=False)

    print("=== Multi-CPR processing complete ===")


process_multi_cpr()
