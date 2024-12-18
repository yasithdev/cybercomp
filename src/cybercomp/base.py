from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, NamedTuple, TypeVar, get_args, get_origin

Step = Any  # NOTE - don't import the real class, as it causes circular imports

T = TypeVar("T", contravariant=True)
RunState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]

# --------------------------------------------
# Base Types
# --------------------------------------------


class Type(Generic[T]):
    """
    Base class for a semantic type

    """


class Category:
    """
    Base class for a category of semantic types

    """


class Model:
    """
    Base class for a Computational Model

    """

    run_engines: list[str]
    run_command: str

    @classmethod
    def describe(cls, level: int = 0, silent: bool = False) -> tuple[set[Parameter], set[Observable]]:
        """
        Describe the model

        """
        rp = set[RequiredParameter]()
        op = set[DefaultParameter]()
        ob = set[Observable]()
        for e in cls.__dict__.values():
            origin = typ = None
            if hasattr(e, "typing"):
                typ = e.typing
            if hasattr(e, "__class__"):
                origin = e.__class__
            if origin is None or typ is None:
                continue
            if issubclass(origin, RequiredParameter):
                rp.add(e)
            if issubclass(origin, DefaultParameter):
                op.add(e)
            if issubclass(origin, Observable):
                ob.add(e)
        prefix = "  " * level
        if not silent:
            print(f"{prefix}* [Model] {cls.__name__}")
            print(f"{prefix}  - params?: {len(op)}", [*map(str, op)])
            print(f"{prefix}  - params!: {len(rp)}", [*map(str, rp)])
            print(f"{prefix}  - observ : {len(ob)}", [*map(str, ob)])
        return {*rp, *op}, ob

    def update(
        self,
        params: set[Parameter],
    ) -> None:
        """
        Update the model with the given parameters, and fill in missing observables

        """
        for p in params:
            setattr(self, p.__class__.__name__, p)
        for key in self.__dir__():
            attr = getattr(self, key)
            if isinstance(attr, Observable) and not attr.initialized:
                attr(f"./{hash(attr)}")


class Engine:
    """
    Base class for a Computational Engine

    """

    @classmethod
    def describe(cls, level: int = 0, silent: bool = False) -> set[Hyperparameter]:
        """
        Describe the engine

        """
        hp = set()
        thp = set[Hyperparameter]()
        for k, v in cls.__dict__.items():
            origin, typ = get_origin(v), get_args(v)
            if origin is None:
                continue
            if issubclass(origin, Hyperparameter):
                hp.add(k)
                thp.add(v)
        prefix = "  " * level
        if not silent:
            print(f"{prefix}* [Engine] {cls.__name__}")
            print(f"{prefix}  - hparams: {len(hp)} >", hp)
        return thp

    def update(
        self,
        hparams: set[Hyperparameter],
    ) -> None:
        """
        Update the engine with the given hyperparameters

        """
        for h in hparams:
            setattr(self, h.__class__.__name__, h)


# --------------------------------------------
# Argument Definitions
# --------------------------------------------


class TypedArgument(Generic[T]):
    """
    Base class for a typed argument

    """

    initialized: bool = False
    typing: TypeVar
    value: T

    def __init__(self, t: TypeVar) -> None:
        super().__init__()
        self.typing = t

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TypedArgument):
            return False
        type_match = self.typing == value.typing
        if self.initialized and value.initialized:
            value_match = self.value == value.value
        else:
            value_match = True
        return type_match and value_match

    def __hash__(self) -> int:
        type_hash = hash(self.typing)
        value_hash = 0
        if self.initialized:
            value_hash = hash(self.value)
        return type_hash + value_hash


class Parameter(TypedArgument[T]):
    """
    Base class for a parameter

    """

    def __call__(self, v: T) -> Parameter:
        self.value = v
        self.initialized = True
        return self

    def __str__(self) -> str:
        if self.initialized:
            return f"P[{self.typing}]({self.value})"
        else:
            return f"P[{self.typing}](not set)"


class RequiredParameter(Parameter[T]):
    """
    Base class for a required parameter

    """


class DefaultParameter(Parameter[T]):
    """
    Base class for a parameter with a default value

    """

    def __init__(self, t: TypeVar, v: T) -> None:
        super().__init__(t)
        self(v)


class Hyperparameter(TypedArgument[T]):
    """
    Base class for a hyperparameter

    """

    def __call__(self, v: T) -> Hyperparameter:
        self.value = v
        self.initialized = True
        return self

    def __str__(self) -> str:
        if self.initialized:
            return f"H[{self.typing}]({self.value})"
        else:
            return f"H[{self.typing}](not set)"


class Observable(TypedArgument[T]):
    """
    Base class for an observation

    """

    def __call__(self, v: T) -> Observable:
        self.value = v
        self.initialized = True
        return self

    def __str__(self) -> str:
        if self.initialized:
            return f"O[{self.typing}]({self.value})"
        else:
            return f"O[{self.typing}](not set)"


# --------------------------------------------
# Execution Types and Definitions
# --------------------------------------------

Args = set[Parameter | Hyperparameter]
Params = set[Parameter]
Hparams = set[Hyperparameter]
Obs = set[Observable]


class RunConfig:
    params: Params
    hparams: Hparams
    obs: Obs

    def __init__(self, params: Params, hparams: Hparams, obs: Obs) -> None:
        self.params = params
        self.hparams = hparams
        self.obs = obs
        self.dir_name = hash(frozenset(params)) + hash(frozenset(hparams)) + hash(frozenset(obs))


Observation = Any
OutSet = set[Observation]

ObsQuery = set[Observable] | None
ObsMap = dict[Observable, Observation]
ObservationSet = set[set[Observable]]


class Runtime(ABC):
    """
    Base class for an execution context

    Objectives:

    1. during __init__(), iterate past runs (in runs_dir/) and populate the execution_log
    2. track each run (key = hash(args,obs); self.history[key] = runs_dir/run_id) for future lookup

    """

    runs_dir: str = "runs/"
    history = {}

    @abstractmethod
    def run(self, step: Step, level: int = 0) -> RunState:
        """
        Run a computational step

        @param step: computational step to run
        @return: the execution state

        """

    @abstractmethod
    def poll(self, step: Step, level: int = 0) -> RunState:
        """
        Poll the execution state of a computational step

        @param step: computational step to run
        @return: the execution state

        """

    @abstractmethod
    def fetch(self, step: Step, query: ObsQuery, level: int = 0) -> ObsMap:
        """
        Fetch observations from a completed computational step

        @param step: computational step to run
        @param query: the observables to fetch
        @return: the observations

        """


class Runnable(ABC):
    """
    Base class for a runnable object

    """

    name: str
    config: RunConfig | None = None

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def describe(self, level: int = 0) -> None:
        """
        Describe the runnable object

        """

    @abstractmethod
    def setup(self, args: Args, level: int = 0) -> None:
        """
        Setup to run for the given arguments

        @param args: the arguments to run with

        """

    @abstractmethod
    def run(self, runtime: Runtime, level: int = 0) -> RunState:
        """
        Execute on a runtime

        @return: the execution state

        """

    @abstractmethod
    def poll(self, runtime: Runtime, level: int = 0) -> RunState:
        """
        Poll for execution state

        @return: the execution state

        """

    @abstractmethod
    def fetch(self, runtime: Runtime, observables: ObsQuery = None, level: int = 0) -> ObsMap:
        """
        Fetch observations for the run, from the runtime

        @param args: the run sets to run
        @return: the fetched observables

        """
