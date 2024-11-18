"""
Loads Type Definitions Given in YAML into Strongly Typed Python Objects

"""

from pathlib import Path
from typing import Any

import black
from tqdm import tqdm
from requests import get

from .codegen import generate_class_py, generate_module
from .specs import EngineSpec, ModelSpec, SourceSpec, TypeSpec, get_canonical_type
from .util_recipes import recipe_to_fs


class API:
    """
    API that connects to running cybercomp server

    """

    def __init__(self, server_url: str, base_path: Path) -> None:
        self.models = dict[str, dict[str, ModelSpec]]()
        self.engines = dict[str, EngineSpec]()
        self.types = dict[str, TypeSpec]()
        self.sources = dict[str, SourceSpec]()
        self.server_url = server_url
        self.out_dir = base_path

    def sync(self) -> None:
        # check for connectivity
        try:
            res = get(f"{self.server_url}/status")
            print(f"Connected to cybercomp server on {self.server_url}")
        except:
            raise ConnectionError(f"Could not connect to cybercomp server on {self.server_url}") from None
        # fetch metadata and generate python stubs
        self.load_types()
        self.load_models()
        self.load_engines()
        self.load_sources()
        print("Completions generated at:", self.out_dir.absolute().as_posix())

    def load_types(self) -> None:
        data: dict[str, Any] = get(f"{self.server_url}/types").json()
        pbar = tqdm(data.items(), desc="[Types] ")
        for key, typ in pbar:
            t = TypeSpec(**typ)
            self.types[key] = t
            pbar.set_postfix_str(key)
        self.generate_types_stubs()

    def load_models(self) -> None:
        data: dict[str, Any] = get(f"{self.server_url}/models").json()
        pbar = tqdm(data.items(), desc="[Models] ")
        for key, model in pbar:
            module, name = key.rsplit("/", maxsplit=1)
            m = ModelSpec(**model)
            m.validate(self.types)
            if module not in self.models:
                self.models[module] = {}
            self.models[module][name] = m
            pbar.set_postfix_str(key)
        self.generate_model_stubs()

    def load_engines(self) -> None:
        data: dict[str, Any] = get(f"{self.server_url}/engines").json()
        pbar = tqdm(data.items(), desc="[Engines] ")
        for key, engine in pbar:
            e = EngineSpec(**engine)
            self.engines[key] = e
            pbar.set_postfix_str(key)
        self.generate_engine_stubs()

    def load_sources(self) -> None:
        data: dict[str, Any] = get(f"{self.server_url}/sources").json()
        pbar = tqdm(data.items(), desc="[Sources] ")
        for key, source in pbar:
            s = SourceSpec(source)
            self.sources[key] = s
            pbar.set_postfix_str(key)

    def generate_types_stubs(self):
        typ_dir = self.out_dir / "types"
        typ_dir.mkdir(parents=True, exist_ok=True)

        # generate a single types stub
        typedefs = dict[str, str]()
        import builtins

        for k, v in self.types.items():
            typ = get_canonical_type(v.type)
            assert hasattr(builtins, typ), f"'{typ}' is not a primitive type"
            typedefs[k] = f"TypeVar('{k}', {typ}, {typ})"

        code = generate_module(imports=[("typing", "TypeVar")], typedefs=typedefs)
        code = black.format_str(code, mode=black.Mode())
        fp = typ_dir / f"__init__.py"
        with open(fp, "w") as f:
            f.write(code)
        # print("[Type] Created", fp.as_posix())

    def generate_model_stubs(self):
        model_dir = self.out_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # generate __init__.py for base module
        fp = model_dir / f"__init__.py"
        code = generate_module(imports=list((f".", k) for k in self.models.keys()))
        code = black.format_str(code, mode=black.Mode())
        with open(fp, "w") as f:
            f.write(code)
        # print("[Module] Created", fp.as_posix())

        # generate independent class files
        for module, models in self.models.items():
            module_dir = model_dir / module
            module_dir.mkdir(parents=True, exist_ok=True)

            # generate __init__.py for module
            fp = module_dir / f"__init__.py"
            code = generate_module(
                imports=list((f".{k}", k) for k in models.keys()),
            )
            code = black.format_str(code, mode=black.Mode())
            with open(fp, "w") as f:
                f.write(code)
            # print("[Module] Created", fp.as_posix())

            # generate class files inside module
            for model_id, model in models.items():
                fp = module_dir / f"{model_id}.py"

                rparams = dict[str, str]()
                dparams = dict[str, str]()
                oparams = dict[str, str]()

                for k, v in model.parameters.items():
                    default = v.default
                    if default is None:
                        rparams[k] = f"RequiredParameter({k})"
                    else:
                        dparams[k] = f"DefaultParameter({k}, {repr(default)})"

                for k in model.observables:
                    oparams[k] = f"Observable({k})"

                # generate run information
                run_engines = model.run.engine
                run_command = model.run.command

                code = generate_class_py(
                    imports=[
                        ("cybercomp", "Model"),
                        ("cybercomp", "RequiredParameter"),
                        ("cybercomp", "DefaultParameter"),
                        ("cybercomp", "Observable"),
                        ("...types", "*"),
                    ],
                    class_name=model_id,
                    class_bases=["Model"],
                    docstring=model.description,
                    fixed_typeless_params={
                        **dparams,
                        **rparams,
                        **oparams,
                        "run_engines": repr(run_engines),
                        "run_command": repr(run_command),
                    },
                )
                code = black.format_str(code, mode=black.Mode())
                with open(fp, "w") as f:
                    f.write(code)
                # print("[Model] Created", fp.as_posix())

    def generate_engine_stubs(self):
        engine_dir = self.out_dir / "engines"
        engine_dir.mkdir(parents=True, exist_ok=True)

        # generate independent class files
        for engine_id, engine in self.engines.items():
            fp = engine_dir / f"{engine_id}.py"

            params = dict[str, str]()
            for k in engine.parameters:
                params[k] = f"Hyperparameter[{k}]"

            code = generate_class_py(
                imports=[
                    ("cybercomp", "Engine"),
                    ("cybercomp", "RequiredParameter"),
                    ("cybercomp", "DefaultParameter"),
                    ("cybercomp", "Observable"),
                    ("cybercomp", "Hyperparameter"),
                    ("..types", "*"),
                ],
                class_name=engine_id,
                class_bases=["Engine"],
                docstring=engine.description,
                fixed_params={},
                fixed_typeless_params=params,
            )
            code = black.format_str(code, mode=black.Mode())
            with open(fp, "w") as f:
                f.write(code)
            # print("[engine] Created", fp.as_posix())

        # generate __init__.py for engine imports
        fp = engine_dir / f"__init__.py"
        code = generate_module(
            imports=list((f".{k}", k) for k in self.engines.keys()),
        )
        with open(fp, "w") as f:
            f.write(code)
        # print("[Module] Created", fp.as_posix())
