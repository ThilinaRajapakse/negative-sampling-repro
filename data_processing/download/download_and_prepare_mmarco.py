import json
import os

import datasets
import pandas as pd
from tqdm.auto import tqdm

split = "dev"

languages = [
    "indonesian",
    "arabic",
    "japanese",
    "russian",
    "chinese",
    "french",
    "dutch",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "hindi",
    "vietnamese",
]

print("=== Downloading MMARCO ===")

for i, language in enumerate(languages):
    print(f"Language: {language}")
    print(f"Language {i+1} of {len(languages)}")
    query_dataset = datasets.load_dataset("unicamp-dl/mmarco", "queries-" + language)

    corpus_dataset = datasets.load_dataset(
        "unicamp-dl/mmarco", "collection-" + language
    )
    print("Loaded corpus of size", len(corpus_dataset["collection"]))
    print("---------------------------------------")

    qrels_df = pd.read_csv(f"../../data/msmarco/{split}.tsv", sep="\t")

    os.makedirs(f"../../data/mmarco/beir/{language}/qrels", exist_ok=True)
    qrels_df.to_csv(
        f"../../data/mmarco/beir/{language}/qrels/{split}.tsv", sep="\t", index=False
    )

    queries_dict = {
        "_id": [],
        "text": [],
    }

    for query in tqdm(query_dataset[split]):
        queries_dict["_id"].append(str(query["id"]))
        queries_dict["text"].append(query["text"])

    query_df = pd.DataFrame(queries_dict)

    query_df.to_json(
        f"../../data/mmarco/beir/{language}/queries.jsonl", orient="records", lines=True
    )

    corpus_dataset = corpus_dataset["collection"]
    corpus_dataset = corpus_dataset.rename_column("id", "_id")
    corpus_dataset = corpus_dataset.cast(
        datasets.Features(
            {"_id": datasets.Value("string"), "text": datasets.Value("string")}
        )
    )

    corpus_dataset.to_json(
        f"../../data/mmarco/beir/{language}/corpus.jsonl", orient="records", lines=True
    )

print("=== MMARCO download complete ===")
