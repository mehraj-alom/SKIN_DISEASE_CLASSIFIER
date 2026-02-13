from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path
    device: str
    filename : str = None

