"""
Bias Evaluation Tool - Proof of Concept
"""
import os
from typing import Protocol, Optional
from openai import OpenAI
from dotenv import load_dotenv
from models import ModelResponse

# Load environment variables
load_dotenv()


class ModelAdapter(Protocol):
    """Interface for model adapters (Dependency Inversion Principle)."""

    def generate(self, prompt: str) -> str:
        """Generate a response from the model."""
        ...


class OpenAIAdapter:
    """OpenAI implementation of ModelAdapter."""

    def __init__(self, model: str = "gpt-5-nano", api_key: Optional[str] = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:
        """Generate a response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def test_connection():
    """Test OpenAI connection with a simple prompt."""
    print("Testing OpenAI connection...")

    adapter = OpenAIAdapter(model="gpt-5-nano")
    test_prompt = "Say 'Hello! Connection successful.' and nothing else."

    response = adapter.generate(test_prompt)
    print(f" Connection successful!")
    print(f"Response: {response}")

    # Create a ModelResponse object to test our data model
    model_response = ModelResponse(
        prompt=test_prompt,
        response_text=response,
        model_name="gpt-5-nano"
    )

    print(f"\n Data model working!")
    print(f"Model response object: {model_response.model_dump()}")


if __name__ == "__main__":
    test_connection()
