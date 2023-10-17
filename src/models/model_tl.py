# Data processing
import pandas as pd
import numpy as np
from src.data.get_and_save_data import *

# Modeling
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

# Hugging Face Dataset
from datasets import Dataset

# Model performance evaluation
import evaluate

def stemming(df, use_stemming):
    if use_stemming:
        df.drop(['Review Text'], axis='columns', inplace=True)
    else:
        df.drop(['Stemmed Review Text'], axis='columns', inplace=True)
    return df

    # Funtion to tokenize data
def tokenize_dataset(data):
    return tokenizer(data["Stemmed Review Text"],
                     max_length=128,
                     truncation=True,
                     padding="max_length")

def preprocess_and_tokenize_data(train_data, test_data):
    # Convert pyhton dataframe to Hugging Face arrow dataset
    hg_train_data = Dataset.from_pandas(train_data)
    hg_test_data = Dataset.from_pandas(test_data)

    # Tokenize the dataset
    dataset_train = hg_train_data.map(tokenize_dataset)
    dataset_test = hg_test_data.map(tokenize_dataset)

    # Remove the review and index columns because it will not be used in the model
    dataset_train = dataset_train.remove_columns(["Stemmed Review Text", "__index_level_0__"])
    dataset_test = dataset_test.remove_columns(["Stemmed Review Text", "__index_level_0__"])

    # Rename label to labels because the model expects the name labels
    dataset_train = dataset_train.rename_column("Top Product", "labels")
    dataset_test = dataset_test.rename_column("Top Product", "labels")

    # Change the format to PyTorch tensors
    dataset_train.set_format("torch")
    dataset_test.set_format("torch")

    # DataLoader
    train_dataloader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=4)
    eval_dataloader = DataLoader(dataset=dataset_test, batch_size=4)

    return train_dataloader, eval_dataloader

def training(train_dataloader, model):
    # Number of epochs
    num_epochs = 2

    # Number of training steps
    num_training_steps = num_epochs * len(train_dataloader)

    # Optimizer
    optimizer = AdamW(params=model.parameters(), lr=5e-6)

    # Set up the learning rate scheduler
    lr_scheduler = get_scheduler(name="linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps)


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

def evaluation(eval_dataloader, model):
    # Load the evaluation metric
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("f1")
    metric3 = evaluate.load("recall")

    # Tells the model that we are evaluting the model performance
    model.eval()

    #  A list for all logits
    logits_all = []

    # A list for all predicted probabilities
    predicted_prob_all = []

    # A list for all predicted labels
    predictions_all = []

    # Loop through the batches in the evaluation dataloader
    for batch in eval_dataloader:
        # Disable the gradient calculation
        with torch.no_grad():
            # Compute the model output
            outputs = model(**batch)
        # Get the logits
        logits = outputs.logits
        # Append the logits batch to the list
        logits_all.append(logits)
        # Get the predicted probabilities for the batch
        predicted_prob = torch.softmax(logits, dim=1)
        # Append the predicted probabilities for the batch to all the predicted probabilities
        predicted_prob_all.append(predicted_prob)
        # Get the predicted labels for the batch
        predictions = torch.argmax(logits, dim=-1)
        # Append the predicted labels for the batch to all the predictions
        predictions_all.append(predictions)
        # Add the prediction batch to the evaluation metric
        metric1.add_batch(predictions=predictions, references=batch["labels"])
        metric2.add_batch(predictions=predictions, references=batch["labels"])
        metric3.add_batch(predictions=predictions, references=batch["labels"])


if __name__ == '__main__':
    path_data = "./data/processed/processed_data.csv"
    df = get_data_from_local(path_data)

    # Set this flag based on whether stemming is applied or not
    use_stemming = True
    df = stemming(df, use_stemming)

    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)

    # Tokenizer from a pretrained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Preprocess and tokenize data
    train_dataloader, eval_dataloader = preprocess_and_tokenize_data(train_data, test_data)

    #Load model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    training(train_dataloader, model)

    evaluation(eval_dataloader, model)
