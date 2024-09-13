from typing import TypeVar
from pathlib import Path
from yaml import safe_load
from pydantic import BaseModel

MI = TypeVar("MI")  # type of model inputs (initial values, params, hparams)
MO = TypeVar("MO")  # type of model observations (trajectories of values, params, hparams)
EH = TypeVar("EH")  # type of engine hyperparameters

data_dir = Path(__file__).parent / "data"


class Type(BaseModel):
    """
    A single semantic type

    """

    formats: list[str]


class TypeGroup(BaseModel):
    """
    A group of semantic types

    """

    category: str
    types: dict[str, Type]


class Engine(BaseModel):
    """
    A computational engine (i.e., the execution program)

    """

    source_id: str
    command: list[str]


class Model(BaseModel):
    """
    A computational model, described by parameters and observations

    """

    description: str
    required_parameters: dict[str, str]
    optional_parameters: dict[str, str]
    observations: dict[str, str]

    def validate(self, type_groups: dict[str, TypeGroup]):
        # flatten the types into dict
        flattened_types: dict[str, list[str]] = dict()
        for group in type_groups.values():
            for typ, spec in group.types.items():
                flattened_types[f"{group.category}/{typ}"] = spec.formats
        # check if required parameters match types
        for param, form in self.required_parameters.items():
            # validate if type is defined
            if param not in flattened_types:
                raise TypeError(f"type={param} does not exist")
            # validate if type spec matches
            if form not in flattened_types[param]:
                raise TypeError(f"format={form} does not match the defined type={param}")
        print("")


class Manager:
    """
    Manager for Cybercomp Objects and Semantic Types

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

        # load type groups
        for fp in data_dir.glob("types/*.type.yml"):
            with open(fp, "r") as f:
                data = safe_load(f)
            key = fp.as_posix()
            g = TypeGroup(**data)
            self.types[g.category] = TypeGroup(**data)
            print("[Type] Loaded", key)

        # load models
        for fp in data_dir.glob("models/*.model.yml"):
            with open(fp, "r") as f:
                data: dict = safe_load(f)
            if data["optional_parameters"] is None:
                data["optional_parameters"] = {}
            key = fp.as_posix()
            m = Model(**data)
            m.validate(self.types)
            self.models[key] = m
            print("[Model] Loaded", key)

        # load engines
        for fp in data_dir.glob("engines/*.engine.yml"):
            with open(fp, "r") as f:
                data = safe_load(f)
            key = fp.as_posix()
            self.engines[key] = Engine(**data)
            print("[Engine] Loaded", key)

        # load sources list
        for fp in data_dir.glob("sources/*"):
            key = fp.as_posix()
            print("[Source] Loaded", key)
            self.sources.add(key)

    @staticmethod
    def check_model_typematch(
        model_inp: MI,
        model_obs: MO,
        candidate_inp: MI,
        candidate_obs: MO,
    ) -> bool:
        """
        right now its doing nothing
        supposed to do type matching based on semantics

        """
        return True

    @staticmethod
    def check_engine_typematch(
        engine_hyp: EH,
        candidate_hyp: EH,
    ) -> bool:
        """
        right now its doing nothing
        supposed to do type matching based on semantics

        """
        return True
