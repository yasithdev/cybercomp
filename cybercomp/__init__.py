from pathlib import Path

lib_dir = Path(__file__).parent
data_dir = lib_dir / "database"
types_dir = lib_dir / "generated" / "types"
model_dir = lib_dir / "generated" / "models"
engine_dir = lib_dir / "generated" / "engines"
types_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
engine_dir.mkdir(parents=True, exist_ok=True)


from .generics import Type, Parameter, Hyperparameter, Observation
from .base import Engine, Model, Runtime
from .experiment import Collection, Experiment
from .generated import engines, models
from .types import Typeshed

__all__ = [
    "Type",
    "Parameter",
    "Hyperparameter",
    "Observation",
    "Engine",
    "Model",
    "Runtime",
    "Experiment",
    "Collection",
    "Typeshed",
    "models",
    "engines",
]
