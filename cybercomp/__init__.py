from pathlib import Path

lib_dir = Path(__file__).parent
data_dir = lib_dir / "database"
model_dir = lib_dir / "generated" / "models"
engine_dir = lib_dir / "generated" / "engines"
model_dir.mkdir(parents=True, exist_ok=True)
engine_dir.mkdir(parents=True, exist_ok=True)


from .base import Engine, Model, Runtime
from .experiment import Collection, Experiment
from .generated import engines, models
from .types import Typeshed

__all__ = [
    "Model",
    "Engine",
    "Experiment",
    "Collection",
    "Runtime",
    "Typeshed",
    "models",
    "engines",
]
