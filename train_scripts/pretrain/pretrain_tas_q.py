import logging

import pandas as pd
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
from multiprocess import set_start_method

# Set up logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Define the path to the training data
train_data_path = "../../data/msmarco/msmarco-train.tsv"

# Load the training data
if train_data_path.endswith(".tsv"):
    train_data = pd.read_csv(train_data_path, sep="\t")
else:
    train_data = train_data_path

# Set up the model arguments
model_args = RetrievalArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_cached_eval_features = False
model_args.include_title = False if "msmarco" in train_data_path else True
model_args.max_seq_length = 256
model_args.num_train_epochs = 40
model_args.train_batch_size = 16
model_args.use_hf_datasets = True
model_args.learning_rate = 1e-6
model_args.warmup_steps = 5000
model_args.save_steps = 300000
model_args.wandb_project = "Negative Sampling Multilingual - Pretrain"
model_args.n_gpu = 1
model_args.data_format = "beir"
model_args.cluster_every_n_epochs = 10
model_args.output_dir = f"../../trained_models/pretrained/TAS-query-msmarco"
model_args.wandb_kwargs = {"name": f"TAS-query-msmarco"}

# Clustering on the queries
model_args.cluster_queries = True
model_args.tas_clustering = True

# Define the model type and names
model_type = "custom"
model_name = None
context_name = "bert-base-multilingual-cased"
question_name = "bert-base-multilingual-cased"

# Main execution
if __name__ == "__main__":
    # Set the start method for multiprocessing
    set_start_method("spawn")

    # Create the model
    model = RetrievalModel(
        model_type,
        model_name,
        context_name,
        question_name,
        args=model_args,
    )

    # Train the model
    model.train_model(
        train_data,
        eval_set="dev",
    )