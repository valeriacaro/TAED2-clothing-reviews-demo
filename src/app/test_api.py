from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from src.app.api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    # Use the TestClient with a `with` statement to trigger the startup and shutdown events.
    with TestClient(app) as client:
        return client


@pytest.fixture
def payload():
    return {
        "sentiment": "No Top",
        "probabilities": 0.6024662256240845,
    }


def test_root(client):
    response = client.get("/")
    json = response.json()
    assert response.status_code == 200
    assert (
        json["data"]["message"]
        == "Welcome to Sentiment Analysis Clothing Reviews! Please, read the `/docs`!"
    )
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["method"] == "GET"
    assert json["url"] == "http://testserver/"
    assert json["timestamp"] is not None



def test_get_one_model(client):
    response = client.get("/model")
    json = response.json()
    assert response.status_code == 200
    assert json["data"] == [
        {
            "type": "BertFinetuned",
            "parameters": {
            "random_state": 2023},
            "accuracy": {"accuracy": 0.7981002871658935},
        }
    ]
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["method"] == "GET"
    assert json["url"] == "http://testserver/model"
    assert json["timestamp"] is not None


def test_get_one_model_not_found(client):
    response = client.get("/model?type=RandomForestClassifier")
    assert response.status_code == 400
    assert response.json()["detail"] == "Type not found"


def test_model_prediction(client, payload):
    response = client.post("/model", json=payload)
    json = response.json()
    assert response.status_code == 200
    assert json["data"]["prediction"] == 2
    assert json["message"] == "OK"
    assert json["status-code"] == 200
    assert json["method"] == "POST"
    assert json["url"] == "http://testserver/model"
    assert json["timestamp"] is not None


def test_model_prediction_not_found(client, payload):
    response = client.post("/model/RandomForestClassifier", json=payload)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] == "Model not found"