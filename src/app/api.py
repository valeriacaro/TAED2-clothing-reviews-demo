from functools import wraps

from fastapi import FastAPI, HTTPException, Request

from typing import List

from http import HTTPStatus
import datetime

from torch import argmax
import torch.nn.functional as F

from torch.utils.data import DataLoader


from src import ROOT_PATH
from src.models.train_model import *
from src.app.schemas import SentimentRequest, SentimentResponse


MODEL_PATH = ROOT_PATH / "model"

model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title="Sentiment Analysis - Clothing Reviews",
    description="This API lets you make predictions on clothing-reviews dataset using bert-finetuned model.",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.on_event("startup")
def _load_models():
    """Loads models"""
    sentiment_model = torch.load(MODEL_PATH / "transfer-learning.pt", map_location="cpu")
    return sentiment_model


@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Sentiment Analysis Clothing Reviews! Please, read the `/docs`!"},
    }
    return response


def _get_models_list(request: Request, type: str = None):
    """Return the list of available models"""
    available_models = [
        {
            "type": model["type"],
            "parameters": model["params"],
            "accuracy": model["metrics"],
        }
        for model in model_wrappers_list
        if model["type"] == type or type is None
    ]

    if not available_models:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Type not found")
    else:
        return {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": available_models,
        }


def preprocess(data) -> Dataset:

    words = data.split()
    data = pd.DataFrame({'Review Text': words})
    # Convert Python DataFrame to Hugging Face arrow dataset
    hg_data = Dataset.from_pandas(data)

    # Tokenize the data sets
    dataset = hg_data.map(tokenize_dataset)
    # Remove the review and index columns because it will not be used in the model
    dataset = dataset.remove_columns(["Review Text"])
    # Change the format to PyTorch tensors
    dataset.set_format("torch")

    return dataset


def predict_sentiment(text: str):
    """."""
    dataset_text = preprocess(text)
    text_dataloader = DataLoader(dataset=dataset_text, shuffle=True, batch_size=4)
    model = _load_models()

    model.eval()

    #  A list for all logits
    logits_all = []

    # A list for all predicted probabilities
    pred_prob_all = []

    # A list for all predicted labels
    predictions_all = []

    # Loop through the batches in the evaluation dataloader
    for batch in text_dataloader:
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
        pred_prob_all.append(predicted_prob)
        # Get the predicted labels for the batch
        prediction = argmax(logits, dim=-1)
        # Append the predicted labels for the batch to all the predictions
        predictions_all.append(prediction)

    class_names = ["No Top", "Top"]

    # Calcular el promedio de las probabilidades para cada etiqueta
    average_probabilities = torch.mean(torch.stack([p[:len(pred_prob_all[-1])] for p in pred_prob_all]), dim=0)

    # Convierte las probabilidades en un diccionario válido
    probabilities_dict = dict(zip(class_names, average_probabilities[0]))

    # Suponiendo que probabilities_dict contiene el diccionario con tensores
    probabilities = probabilities_dict
    sentiment = max(probabilities, key=probabilities.get)  # Obtiene la clave con el valor máximo
    probability_value = probabilities[sentiment].item()  # Obtiene el valor correspondiente a la clave

    print(f"sentiment = {sentiment}, probabilities = {probability_value}")

    return sentiment, probability_value


@app.post("/predict", response_model=SentimentResponse)
def _predict(request: SentimentRequest):
    try:
        sentiment, prob = predict_sentiment(request.text)
        return SentimentResponse(sentiment=sentiment, probabilities=prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
