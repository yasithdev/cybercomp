"""
Loads Type Definitions Given in YAML into Strongly Typed Python Objects

"""

from pathlib import Path

from yaml import safe_load

from .codegen import FS, generate_class_py, generate_init_py
from .specs import EngineSpec, ModelSpec, TypeGroupSpec, RecipeSpec


def recipe_to_fs(recipe: RecipeSpec) -> FS:
    args = {}
    command: list[str] = []
    for chunk in recipe.command:
        if ("[@p:" in chunk or "[@o:" in chunk) and "]" in chunk:
            i, j = chunk.index("[@"), chunk.index("]")
            arg = chunk[i + 4 : j]
            args[arg] = "str"
        command.append(chunk.replace(f"[@p:", f"[@").replace(f"[@o:", f"[@"))
    for arg in args.keys():
        for chunk in command:
            chunk = chunk.replace(f"[@p:{arg}]", f"[@{arg}]").replace(f"[@o:{arg}]", f"[@{arg}]")
    return FS(command=command, args=args, rtype="list[str]")


class Typeshed:
    """
    Type Definition Generator.
    Creates objects for each definition and stores them for autocompletion

    """

    models: dict[str, ModelSpec]
    types: dict[str, TypeGroupSpec]
    engines: dict[str, EngineSpec]
    sources: set[str]

    def __init__(self) -> None:
        self.models = dict()
        self.engines = dict()
        self.types = dict()
        self.sources = set()
        self.data_dir = Path(__file__).parent / "database"

    def sync(self) -> None:

        self.load_types()
        self.load_models()
        self.load_engines()
        self.load_sources()

    def load_types(self) -> None:
        for fp in self.data_dir.glob("types/*.yml"):
            with open(fp, "r") as f:
                data = safe_load(f)
            key = fp.stem
            g = TypeGroupSpec(**data)
            self.types[g.category] = TypeGroupSpec(**data)
            print("[Type] Loaded", key)

    def load_models(self) -> None:
        for fp in self.data_dir.glob("models/*.yml"):
            with open(fp, "r") as f:
                data: dict = safe_load(f)
            if data["optional_parameters"] is None:
                data["optional_parameters"] = {}
            key = fp.stem
            m = ModelSpec(**data)
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
            self.engines[key] = EngineSpec(**data)
            print("[Engine] Loaded", key)
        self.generate_engine_code()

    def load_sources(self) -> None:
        for fp in self.data_dir.glob("sources/*"):
            key = fp.stem
            print("[Source] Loaded", key)
            self.sources.add(key)

    def generate_model_code(self):
        model_dir = self.data_dir.parent / "generated" / "models"

        # generate independent class files
        for model_id, model in self.models.items():
            fp = model_dir / f"{model_id}.py"
            code = generate_class_py(
                imports=[("cybercomp", "Model")],
                class_name=model_id,
                class_bases=["Model"],
                docstring=model.description,
                fixed_params={},
                required_params=list(model.required_parameters.keys()),
                # hardcoded paramtypes to str for now
                required_paramtypes=["str" for _ in model.required_parameters.values()],
                # optional params have None as the default value
                optional_params={k: None for k in model.optional_parameters.keys()},
                # hardcoded paramtypes to str|None for now
                optional_paramtypes=["str | None" for _ in model.optional_parameters.values()],
                functions={},
            )
            with open(fp, "w") as f:
                f.write(code)
            print("[Model] Created", fp.relative_to(self.data_dir.parent).as_posix())

        # generate __init__.py for model imports
        fp = model_dir / f"__init__.py"
        code = generate_init_py(
            imports=list(self.models.keys()),
        )
        with open(fp, "w") as f:
            f.write(code)
        print("[Module] Created", fp.relative_to(self.data_dir.parent).as_posix())

    def generate_engine_code(self):
        engine_dir = self.data_dir.parent / "generated" / "engines"

        # generate independent class files
        for engine_id, engine in self.engines.items():
            fp = engine_dir / f"{engine_id}.py"
            code = generate_class_py(
                imports=[("cybercomp", "Engine")],
                class_name=engine_id,
                class_bases=["Engine"],
                docstring=engine.description,
                fixed_params={"source_id": ("str", engine.source_id)},
                required_params=list(engine.engine_parameters.keys()),
                # hardcoded paramtypes to str for now
                required_paramtypes=["str" for _ in engine.engine_parameters.values()],
                # optional params have None as the default value
                optional_params={},
                # hardcoded paramtypes to str|None for now
                optional_paramtypes=[],
                functions={k: recipe_to_fs(v) for k, v in engine.supported_models.items()},
            )
            with open(fp, "w") as f:
                f.write(code)
            print("[engine] Created", fp.relative_to(self.data_dir.parent).as_posix())

        # generate __init__.py for engine imports
        fp = engine_dir / f"__init__.py"
        code = generate_init_py(
            imports=list(self.engines.keys()),
        )
        with open(fp, "w") as f:
            f.write(code)
        print("[Module] Created", fp.relative_to(self.data_dir.parent).as_posix())
