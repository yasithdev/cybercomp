"""
Loads Type Definitions Given in YAML into Strongly Typed Python Objects

"""

from pathlib import Path

from yaml import safe_load

from .codegen import generate_class, generate_init_py
from .cybercomp import Engine, Model, TypeGroup


class Loader:
    """
    Type Definition Loader.
    Creates objects for each definition and stores them for autocompletion

    """

    models: dict[str, Model]
    types: dict[str, TypeGroup]
    engines: dict[str, Engine]
    sources: set[str]

    def __init__(self) -> None:
        self.models = dict()
        self.engines = dict()
        self.types = dict()
        self.sources = set()
        self.data_dir = Path(__file__).parent / "database"

    def load(self) -> None:
        self.load_types()
        self.load_models()
        self.load_engines()
        self.load_sources()

    def load_types(self) -> None:
        for fp in self.data_dir.glob("types/*.yml"):
            with open(fp, "r") as f:
                data = safe_load(f)
            key = fp.stem
            g = TypeGroup(**data)
            self.types[g.category] = TypeGroup(**data)
            print("[Type] Loaded", key)

    def load_models(self) -> None:
        for fp in self.data_dir.glob("models/*.yml"):
            with open(fp, "r") as f:
                data: dict = safe_load(f)
            if data["optional_parameters"] is None:
                data["optional_parameters"] = {}
            key = fp.stem
            m = Model(**data)
            m.validate(self.types)
            self.models[key] = m
            print("[Model] Loaded", key)
        self.generate_model_code()

    def load_engines(self) -> None:
        for fp in self.data_dir.glob("engines/*.yml"):
            with open(fp, "r") as f:
                data = safe_load(f)
            if data["engine_parameters"] is None:
                data["engine_parameters"] = {}
            key = fp.stem
            self.engines[key] = Engine(**data)
            print("[Engine] Loaded", key)

    def load_sources(self) -> None:
        for fp in self.data_dir.glob("sources/*"):
            key = fp.stem
            print("[Source] Loaded", key)
            self.sources.add(key)

    def generate_model_code(self):
        model_dir = self.data_dir.parent / "generated" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # generate independent class files
        for model_id, model in self.models.items():
            fp = model_dir / f"{model_id}.py"
            code = generate_class(
                class_name=model_id,
                docstring=model.description,
                required_params=list(model.required_parameters.keys()),
                # hardcoded paramtypes to str for now
                required_paramtypes=["str" for _ in model.required_parameters.values()],
                # optional params have None as the default value
                optional_params={k: None for k in model.optional_parameters.keys()},
                # hardcoded paramtypes to str|None for now
                optional_paramtypes=["str | None" for _ in model.optional_parameters.values()],
            )
            with open(fp, "w") as f:
                f.write(code)
            print("[Model] Saved to", fp.relative_to(self.data_dir.parent).as_posix())

        # generate __init__.py for model imports
        fp = model_dir / f"__init__.py"
        code = generate_init_py(
            imports=list(self.models.keys()),
        )
        with open(fp, "w") as f:
            f.write(code)
        print("[Module] Saved to", fp.relative_to(self.data_dir.parent).as_posix())
