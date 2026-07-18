from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class WeatherToolInput(BaseModel):
    location: str = Field(description="Name of the city, for example Kolkata or Paris")
