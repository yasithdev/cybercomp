"""
Loads Type Definitions Given in YAML into Strongly Typed Python Objects

"""

from pathlib import Path
from typing import Any

from requests import get

from .codegen import generate_class_py, generate_module
from .specs import EngineSpec, ModelSpec, SourceSpec, TypeSpec

primitives = dict(
    file="str",
    binary="str",
    archive="str",
)


class Completions:
    """
    Autocompletion Generator that generates Python objects on-demand

    """

    def __init__(self, server_url: str, base_path: Path) -> None:
        self.models = dict[str, ModelSpec]()
        self.engines = dict[str, EngineSpec]()
        self.types = dict[str, TypeSpec]()
        self.sources = dict[str, SourceSpec]()
        self.server_url = server_url
        self.out_dir = base_path

    def sync(self) -> None:
        # fetch metadata and generate python stubs

        self.load_types()
        self.load_models()
        self.load_engines()
        self.load_sources()

    def load_types(self) -> None:
        data: dict[str, Any] = get(f"{self.server_url}/types").json()
        for key, typ in data.items():
            print("[Type] Loaded", key)
            t = TypeSpec(**typ)
            self.types[key] = t
        self.generate_types_stubs()

    def load_models(self) -> None:
        data: dict[str, Any] = get(f"{self.server_url}/models").json()
        for key, model in data.items():
            print("[Model] Loaded", key)
            m = ModelSpec(**model)
            m.validate(self.types)
            self.models[key] = m
        self.generate_model_stubs()

    def load_engines(self) -> None:
        data: dict[str, Any] = get(f"{self.server_url}/engines").json()
        for key, engine in data.items():
            print("[Engine] Loaded", key)
            e = EngineSpec(**engine)
            self.engines[key] = e
        self.generate_engine_stubs()

    def load_sources(self) -> None:
        data: dict[str, Any] = get(f"{self.server_url}/sources").json()
        for key, source in data.items():
            print("[Source] Loaded", key)
            s = SourceSpec(source)
            self.sources[key] = s

    def generate_types_stubs(self):
        typ_dir = self.out_dir / "types"
        typ_dir.mkdir(parents=True, exist_ok=True)

        # generate a single types stub
        code = generate_module(imports=[], typedefs={k: primitives[v.form] for k, v in self.types.items()})
        fp = typ_dir / f"__init__.py"
        with open(fp, "w") as f:
            f.write(code)
        print("[Type] Created", fp.as_posix())

    def generate_model_stubs(self):
        model_dir = self.out_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # generate independent class files
        for model_id, model in self.models.items():
            fp = model_dir / f"{model_id}.py"
            code = generate_class_py(
                imports=[("cybercomp", "Model"), ("cybercomp", "Parameter"), ("cybercomp", "Observation")],
                class_name=model_id,
                class_bases=["Model"],
                docstring=model.description,
                fixed_params={},
                typed_params={k: f"Observation[{primitives[v]}]" for k, v in model.observations.items()},
                required_params=list(model.required_parameters.keys()),
                # hardcoded paramtypes to str for now
                required_paramtypes=[f"Parameter[{primitives[v]}]" for v in model.required_parameters.values()],
                # optional params have None as the default value
                optional_params={k: None for k in model.optional_parameters.keys()},
                # hardcoded paramtypes to str|None for now
                optional_paramtypes=[f"Parameter[{primitives[v]}] | None" for v in model.optional_parameters.values()],
                functions={},
            )
            with open(fp, "w") as f:
                f.write(code)
            print("[Model] Created", fp.as_posix())

        # generate __init__.py for model imports
        fp = model_dir / f"__init__.py"
        code = generate_module(
            imports=list(self.models.keys()),
        )
        with open(fp, "w") as f:
            f.write(code)
        print("[Module] Created", fp.as_posix())

    def generate_engine_stubs(self):
        engine_dir = self.out_dir / "engines"
        engine_dir.mkdir(parents=True, exist_ok=True)

        # generate independent class files
        for engine_id, engine in self.engines.items():
            fp = engine_dir / f"{engine_id}.py"
            code = generate_class_py(
                imports=[
                    ("cybercomp", "Engine"),
                    ("cybercomp", "Parameter"),
                    ("cybercomp", "Observation"),
                    ("cybercomp", "Hyperparameter"),
                ],
                class_name=engine_id,
                class_bases=["Engine"],
                docstring=engine.description,
                fixed_params={"source_id": ("str", engine.source_id)},
                typed_params={},
                required_params=list(engine.engine_parameters.keys()),
                # hardcoded paramtypes to str for now
                required_paramtypes=[f"Hyperparameter[{primitives[v]}]" for v in engine.engine_parameters.values()],
                # optional params have None as the default value
                optional_params={},
                # hardcoded paramtypes to str|None for now
                optional_paramtypes=[],
                functions={k: recipe_to_fs(v) for k, v in engine.supported_models.items()},
            )
            with open(fp, "w") as f:
                f.write(code)
            print("[engine] Created", fp.as_posix())

        # generate __init__.py for engine imports
        fp = engine_dir / f"__init__.py"
        code = generate_module(
            imports=list(self.engines.keys()),
        )
        with open(fp, "w") as f:
            f.write(code)
        print("[Module] Created", fp.as_posix())
