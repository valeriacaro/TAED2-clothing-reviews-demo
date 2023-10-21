from train_model import *
import evaluate
from pathlib import Path
from src import ROOT_PATH
import json


def evaluation(eval_dataloader, model):
    """

    Args:
        eval_dataloader (DataLoader): A DataLoader containing testing data.
        model (transformers.BertForSequenceClassification): The machine learning model to be evaluated.

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

    model = model.to('cpu')

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
    return metric1.compute()


def score_function(eval_dataloader, model):
    """

        Args:
            eval_dataloader (DataLoader): A DataLoader containing testing data.
            model (transformers.BertForSequenceClassification): The machine learning model to be evaluated.

        Returns:
            None

        """
    # Path to the metrics folder
    Path("metrics").mkdir(exist_ok=True)
    metrics_folder_path = ROOT_PATH / "metrics"
    accuracy = evaluation(eval_dataloader, model)
    # Create a dictionary to store the accuracy
    accuracy_dict = {"accuracy": accuracy}
    with open(metrics_folder_path / "scores.json", "w") as scores_file:
        json.dump(accuracy_dict, scores_file, indent=4)
    print("Evaluation completed.")


if __name__ == '__main__':

    pass
