"""
Loads Type Definitions from YAML

"""

from pathlib import Path
from typing import Any

from cybercomp.specs import CategorySpec, EngineSpec, ModelSpec, TypeSpec, SourceSpec
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
    models_dir = db_dir / "models"
    engines_dir = db_dir / "engines"
    sources_dir = db_dir / "sources"

    def __init__(self) -> None:
        # ensure directories exist
        self.engines_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)

    def get_all_types(self) -> dict[str, TypeSpec]:
        models = self.get_all_models()
        engines = self.get_all_engines()
        types = dict[str, TypeSpec]()
        # TODO add key as prefix to each type
        for key, v in models.items():
            types.update(v.parameters)
            types.update(v.observables)
        for key, v in engines.items():
            types.update(v.parameters)
        # for fp in self.types_dir.glob("*.yml"):
        #     key = fp.stem
        #     print("[Type]", key)
        #     c = CategorySpec(**{k: TypeSpec(**v) for k, v in load_yaml(fp).items()})
        #     # prefix each type with its category
        #     for k, v in c.items():
        #         types[f"{key}_{k}"] = v
        return types

    def get_all_models(self) -> dict[str, ModelSpec]:
        models = dict[str, ModelSpec]()
        for fp in self.models_dir.glob("*/*.yml"):
            key = f"{fp.parent.stem}/{fp.stem}"
            fp.parent.stem
            print("[Model]", key)
            m = ModelSpec(**load_yaml(fp))
            models[key] = m
        return models

    def get_all_engines(self) -> dict[str, EngineSpec]:
        engines = dict[str, EngineSpec]()
        for fp in self.engines_dir.glob("*.yml"):
            key = fp.stem
            print("[Engine]", key)
            e = EngineSpec(**load_yaml(fp, dict(parameters={})))
            engines[key] = e
        return engines

    def list_all_sources(self) -> dict[str, SourceSpec]:
        sources = dict[str, str]()
        for fp in self.sources_dir.glob("*"):
            key = fp.stem
            print("[Source]", key)
            s = SourceSpec(fp.as_posix())
            sources[key] = s
        return sources
