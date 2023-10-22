'''
This module is created to store the function that
gives predictions as output to decide if a
product is a top product or not
depending on the review given as input
'''

import evaluate
import torch
from torch import argmax
import torch.nn.functional as F

def prediction(eval_dataloader, model):
    """

    Args:
        eval_dataloader (DataLoader):
        - A DataLoader containing testing data.
        model (transformers.BertForSequenceClassification):
        - The machine learning model to be evaluated.

    Returns:
        None

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

    print(predictions_all)


if __name__ == '__main__':
    pass
