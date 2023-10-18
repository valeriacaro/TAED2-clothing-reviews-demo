""" FastAPI application and three endpoints """
from http import HTTPStatus
from typing import Dict
from datetime import datetime
from functools import wraps
from fastapi import FastAPI, Request
from pathlib import Path
from config import logger
from tagifai import main
from src import MODELS_PATH

# Define application
app = FastAPI(
    title="Sentiment Analysis - Clothing Reviews",
    description="Classify machine learning clothing reviews.",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap

@app.get("/")
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "This is the Clothing Reviews API"},
    }
    return response


@app.on_event("startup")
def load_artifacts():
    """

    """

    run_id = open(Path(config.MODELS_PATH, "run_id.txt")).read()
    artifacts = main.load_artifacts(model_dir=config.MODEL_DIR)
    logger.info("Ready for inference!")