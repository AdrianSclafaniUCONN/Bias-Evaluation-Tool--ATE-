"""
Core data models for bias evaluation framework.
Lambda-ready: These models work locally and in AWS Lambda.
"""
from pydantic import BaseModel
from typing import Dict, Optional, List
from datetime import datetime


class PromptVariation(BaseModel):
    """A single prompt with demographic treatments applied."""
    template: str
    filled_prompt: str
    treatments: Dict[str, str]  # {"race": "Black person", "gender": "woman"}
    run_number: int  # Which repetition this is (for n_runs)


class ModelResponse(BaseModel):
    """Response from a model being evaluated."""
    prompt: str
    response_text: str
    model_name: str
    timestamp: datetime = datetime.now()
    treatments: Dict[str, str] = {}  # {"race": "Black", "gender": "woman"}

    class Config:
        arbitrary_types_allowed = True


class BiasScore(BaseModel):
    """Score from judge model evaluating bias in a response."""
    dimension: str  # e.g., "warmth_bias", "competence_bias"
    score: float  # 0-50 scale (0 = highly biased, 50 = no bias)
    reasoning: str  # Judge's explanation
    response_text: str  # The text that was judged
    treatments: Dict[str, str]  # Demographic variables from the prompt
