import pydantic
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionConfig(BaseModel):
    device: str = "cpu"


