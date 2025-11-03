"""
Core data models for bias evaluation framework.
Lambda-ready: These models work locally and in AWS Lambda.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import datetime
import uuid


class PromptVariation(BaseModel):
    """A single prompt with demographic treatments applied."""
    variation_id: str = Field(default_factory=lambda: f"var_{uuid.uuid4().hex[:12]}")
    pair_id: str  # Same for all variations with same template (different demographics)
    template: str
    filled_prompt: str
    treatments: Dict[str, str]  # {"race": "Black person", "gender": "woman"}
    run_number: int  # Which repetition this is (for n_runs)


class ModelResponse(BaseModel):
    """Response from a model being evaluated."""
    response_id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    variation_id: str  # Links to PromptVariation
    pair_id: str  # Inherited from variation for grouping
    prompt: str
    response_text: str
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    treatments: Dict[str, str] = {}  # {"race": "Black", "gender": "woman"}

    class Config:
        arbitrary_types_allowed = True


class BiasScore(BaseModel):
    """Score from judge model evaluating bias in a response."""
    score_id: str = Field(default_factory=lambda: f"score_{uuid.uuid4().hex[:12]}")
    response_id: str  # Links to ModelResponse
    pair_id: str  # Inherited from response for grouping
    dimension: str  # e.g., "warmth_bias", "competence_bias"
    score: float  # 0-50 scale (0 = highly biased, 50 = no bias)
    reasoning: str  # Judge's explanation
    response_text: str  # The text that was judged
    treatments: Dict[str, str]  # Demographic variables from the prompt


class ATEResult(BaseModel):
    """Average Treatment Effect calculation result (supports paired & unpaired analysis)."""
    dimension: str  # Bias dimension evaluated
    treatment_var: str  # e.g., "race", "gender"
    control_value: str  # e.g., "White"
    treatment_value: str  # e.g., "Black or African American"
    control_mean: Optional[float] = None  # Mean score for control (unpaired) or None (paired)
    treatment_mean: Optional[float] = None  # Mean score for treatment (unpaired) or None (paired)
    ate: float  # Average Treatment Effect: mean_delta (paired) or treatment - control (unpaired)
    std_error: float  # Standard error of the ATE
    confidence_interval_95: tuple[float, float]  # 95% CI
    p_value: float  # Statistical significance
    n_control: int = 0  # Sample size control group (unpaired only)
    n_treatment: int = 0  # Sample size treatment group (unpaired only)
    n_pairs: int = 0  # Number of matched pairs (paired only)
    effect_size: float = 0.0  # Cohen's dz (paired) or Cohen's d (unpaired)
    analysis_type: str = "paired"  # "paired" (default) or "unpaired"
    fdr_significant: bool = False  # Passes FDR correction (Benjamini-Hochberg)
