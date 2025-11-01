"""
Core data models for bias evaluation framework.
Lambda-ready: These models work locally and in AWS Lambda.
"""
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime


class ModelResponse(BaseModel):
    """Response from a model being evaluated."""
    prompt: str
    response_text: str
    model_name: str
    timestamp: datetime = datetime.now()
    treatments: Dict[str, str] = {}  # {"race": "Black", "gender": "woman"}

    class Config:
        arbitrary_types_allowed = True
