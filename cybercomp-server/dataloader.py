"""
Loads Type Definitions from YAML

"""

from pathlib import Path
from typing import Any

from cybercomp.specs import CategorySpec, EngineSpec, ModelSpec, TypeSpec
from yaml import safe_load


def load_yaml(fp: Path, defaults: dict[str, Any] = {}) -> Any:
    with open(fp, "r") as f:
        data = safe_load(f)
    for k, v in defaults.items():
        if k in data and data[k] is None:
            data[k] = v
    return data


class DataLoader:
    """
    Loader for Cybercomp Data Types.

    """

    db_dir = Path(__file__).parent / "database"
    types_dir = db_dir / "types"
    models_dir = db_dir / "models"
    engines_dir = db_dir / "engines"
    sources_dir = db_dir / "sources"

    def __init__(self) -> None:
        # ensure directories exist
        self.engines_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.types_dir.mkdir(parents=True, exist_ok=True)

    def get_all_types(self) -> dict[str, TypeSpec]:
        types = dict[str, TypeSpec]()
        for fp in self.types_dir.glob("*.yml"):
            key = fp.stem
            print("[Type]", key)
            c = CategorySpec(**load_yaml(fp))
            # prefix each type with its category
            for k, v in c.items():
                types[f"{key}_{k}"] = v
        return types

    def get_all_models(self) -> dict[str, ModelSpec]:
        types = self.get_all_types()
        models = dict[str, ModelSpec]()
        for fp in self.models_dir.glob("*.yml"):
            key = fp.stem
            print("[Model]", key)
            m = ModelSpec(**load_yaml(fp, dict(optional_parameters={})))
            m.validate(types)
            models[key] = m
        return models

    def get_all_engines(self) -> dict[str, EngineSpec]:
        engines = dict[str, EngineSpec]()
        for fp in self.engines_dir.glob("*.yml"):
            key = fp.stem
            print("[Engine]", key)
            e = EngineSpec(**load_yaml(fp, dict(engine_parameters={})))
            engines[key] = e
        return engines

    def list_all_sources(self) -> dict[str, str]:
        sources = dict[str, str]()
        for fp in self.sources_dir.glob("*"):
            key = fp.stem
            print("[Source]", key)
            sources[key] = fp.as_posix()
        return sources
