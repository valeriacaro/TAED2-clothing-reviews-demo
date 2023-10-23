"""Definitions for the objects used by our resource endpoints."""

from enum import Enum

from pydantic import BaseModel


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):

    sentiment: str
    probabilities: float


class SentimentType(Enum):
    NoTop = 0
    TopProduct = 1
