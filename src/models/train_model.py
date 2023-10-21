# IMPORTS
# Data processing
import pandas as pd

# Modeling
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler

# Hugging Face Dataset
from datasets import Dataset


# FUNCTIONS
def stemming(df, stem=True) -> pd.DataFrame:
    """
    Selects features based on the stemming flag (stemmed text or not).
    Args:
        df (pd.DataFrame): The input DataFrame.
        stem (bool): A flag indicating whether stemming is applied.

    Returns: A pd.DataFrame containing the feature variable (Review Text or Stemmed Review Text)
            and the target variable (Top Product).

    """
    if stem:
        df.drop(['Review Text'], axis='columns', inplace=True)
    else:
        df.drop(['Stemmed Review Text'], axis='columns', inplace=True)
    return df


def tokenize_dataset(data) -> Dataset:
    """
    Tokenizes "Review Text" data in a Hugging Face arrow dataset using a specified tokenizer.

    Args:
        data (datasets.arrow_dataset.Dataset): The input Hugging Face arrow dataset.

    Returns:
        datasets.arrow_dataset.Dataset: Data tokenized.

    """
    # Tokenizer from a pretrained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    return tokenizer(data["Review Text"], max_length=128, truncation=True, padding="max_length")


def tokenize_dataset_stem(data) -> Dataset:
    """
    Tokenizes "Stemmed Review Text" data in a Hugging Face arrow dataset using a specified tokenizer.

    Args:
        data (datasets.arrow_dataset.Dataset): the input Hugging Face arrow dataset.

    Returns:
        datasets.arrow_dataset.Dataset: data tokenized.

    """
    # Tokenizer from a pretrained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    return tokenizer(data["Stemmed Review Text"], max_length=128, truncation=True, padding="max_length")


def preprocess_and_tokenize_data(data, stem=True) -> Dataset:
    """
    Preprocess and tokenize the input data.
    Args:
        data (pd.DataFrame): The input DataFrame.
        stem (bool): A flag indicating whether stemming is applied.

    Returns:
        dataset (Dataset): The input data in a PyTorch tensor format.

    """
    # Convert Python DataFrame to Hugging Face arrow dataset
    hg_data = Dataset.from_pandas(data)
    # Tokenize the data sets
    if stem:
        dataset = hg_data.map(tokenize_dataset_stem)
        # Remove the review and index columns because it will not be used in the model
        dataset = dataset.remove_columns(["Stemmed Review Text"])
    else:
        dataset = hg_data.map(tokenize_dataset)
        # Remove the review and index columns because it will not be used in the model
        dataset = dataset.remove_columns(["Review Text"])

    # Rename label to labels because the model expects the name labels
    dataset = dataset.rename_column("Top Product", "labels")

    # Change the format to PyTorch tensors
    dataset.set_format("torch")

    return dataset


def training(train_dataloader, model):
    """
    Train a machine learning model using the provided DataLoader and model.
    Args:
        train_dataloader (DataLoader): A DataLoader containing training data.
        model (transformers.BertForSequenceClassification): The machine learning model to be trained.

    Returns:
        None

    """
    # Number of epochs
    num_epochs = 3

    # Number of training steps
    num_training_steps = num_epochs * len(train_dataloader)

    # Optimizer
    optimizer = AdamW(params=model.parameters(), lr=5e-6)

    # Set up the learning rate scheduler
    lr_scheduler = get_scheduler(name="linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    # Use GPU if it is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Tells the model that we are training the model
    model.train()
    # Loop through the epochs
    for epoch in range(num_epochs):
        # Loop through the batches
        for batch in train_dataloader:
            # Compute the model output for the batch
            outputs = model(**batch)
            # Loss computed by the model
            loss = outputs.loss
            # backpropagates the error to calculate gradients
            loss.backward()
            # Update the model weights
            optimizer.step()
            # Learning rate scheduler
            lr_scheduler.step()
            # Clear the gradients
            optimizer.zero_grad()


# MAIN
if __name__ == '__main__':

    pass
