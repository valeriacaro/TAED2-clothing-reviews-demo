"""
This module is created to perform the test
process on our model using the test data.
It also computes some metrics. It can be used
only for getting predictions too
"""

import json
from pathlib import Path
import evaluate
import mlflow
import torch
from torch import argmax
import torch.nn.functional as F
from src import ROOT_PATH


def prediction(eval_dataloader, model, test=False):
    """

    Args:
        eval_dataloader (DataLoader):
        - A DataLoader containing testing data.
        model (transformers.BertForSequenceClassification):
        - The machine learning model to be evaluated.
        test (Boolean variable):
        - True if we want to get test metrics
        - False if we want to make predictions

    Returns:
        - Computed accuracy if test=True
        - Predictions otherwise

    """
    # Load the evaluation metric
    metric1 = evaluate.load("accuracy")

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
        predicted_prob = F.softmax(logits, dim=1)
        # Append the predicted probabilities for the batch to
        # all the predicted probabilities
        predicted_prob_all.append(predicted_prob)
        # Get the predicted labels for the batch
        predictions = argmax(logits, dim=-1)
        # Append the predicted labels for the batch to all the predictions
        predictions_all.append(predictions)
        # Add the prediction batch to the evaluation metric
        metric1.add_batch(
            predictions=predictions, references=batch["labels"]
        )

    if test:
        return metric1.compute()

    return predictions_all


def score_function(eval_dataloader, model):
    """

        Args:
            eval_dataloader (DataLoader):
            - A DataLoader containing testing data.
            model (transformers.BertForSequenceClassification):
            - The machine learning model to be evaluated.

        Returns:
            None

        """
    # Path to the metrics folder
    Path("metrics").mkdir(exist_ok=True)
    metrics_folder_path = ROOT_PATH / "metrics"
    accuracy = prediction(eval_dataloader, model, test=True)
    mlflow.log_metrics(accuracy)
    # Create a dictionary to store the accuracy
    accuracy_dict = {"accuracy": accuracy}
    with open(
            metrics_folder_path / "scores.json", "w", encoding='utf-8'
    ) as scores_file:
        json.dump(accuracy_dict, scores_file, indent=4)
    print("Evaluation completed.")


if __name__ == '__main__':

    pass
