import logging
import pandas as pd
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

# Setting up the logging configuration
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Path to the training data
train_data_path = "../../data/msmarco/msmarco-train.tsv"

# Reading the training data from a TSV file or using the provided path
if train_data_path.endswith(".tsv"):
    train_data = pd.read_csv(train_data_path, sep="\t")
else:
    train_data = train_data_path

# Setting up the model arguments
model_args = RetrievalArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.use_cached_eval_features = False
model_args.include_title = False if "msmarco" in train_data_path else True
model_args.max_seq_length = 256
model_args.num_train_epochs = 40
model_args.train_batch_size = 8
model_args.use_hf_datasets = True
model_args.learning_rate = 1e-6
model_args.warmup_steps = 5000
model_args.save_steps = 300000
model_args.evaluate_during_training = False
model_args.save_model_every_epoch = False

# Setting up the project name for Weights & Biases integration. Remove this line if you don't use W&B.
model_args.wandb_project = "Negative Sampling Multilingual - Pretrain"

# Enabling hard negatives for training
model_args.hard_negatives = True

# Setting up the number of GPUs to use and the data format
model_args.n_gpu = 1
model_args.data_format = "beir"

# Disabling ANCE training
model_args.ance_training = False

# Setting up the model type, model name, context name, and question name
model_type = "custom"
model_name = None
context_name = "bert-base-multilingual-cased"
question_name = "bert-base-multilingual-cased"

# Setting up the Weights & Biases run name
model_args.wandb_kwargs = {"name": f"DPR-BM-msmarco"}

# Setting up the output directory for saving the trained model
model_args.output_dir = f"../../trained_models/pretrained/DPR-BM-msmarco"


# Main execution
if __name__ == "__main__":
    # Setting the start method for multiprocessing
    from multiprocess import set_start_method
    set_start_method("spawn")

    # Create a TransformerModel
    model = RetrievalModel(
        model_type,
        model_name,
        context_name,
        question_name,
        args=model_args,
    )

    # Training the model
    model.train_model(
        train_data,
        eval_set="dev",
    )
