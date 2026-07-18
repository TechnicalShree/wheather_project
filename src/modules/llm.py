"""OpenRouter client and LLM call helpers: basic, JSON mode, structured output."""
import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

MODEL_NAME = "openai/gpt-4o-mini"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


class ConceptExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(description="Name of the concept being explained")
    summary: str = Field(description="A concise explanation in beginner-friendly language")
    difficulty: Literal["beginner", "intermediate", "advanced"]
    key_points: list[str] = Field(
        min_length=2,
        description="Important ideas the learner should remember",
    )
    example: str = Field(description="One practical example")
    confidence: float = Field(ge=0, le=1, description="Confidence score between 0 and 1")


def basic_completion(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful AI instructor. Explain concepts simply."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=300,
    )
    return response.choices[0].message.content


def json_completion(prompt: str, keys_hint: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": f"Return only a valid JSON object. {keys_hint}"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def structured_completion(prompt: str) -> ConceptExplanation:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an AI instructor. Follow the supplied JSON schema exactly."},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "concept_explanation",
                "strict": True,
                "schema": ConceptExplanation.model_json_schema(),
            },
        },
        temperature=0,
    )
    return ConceptExplanation.model_validate_json(response.choices[0].message.content)
