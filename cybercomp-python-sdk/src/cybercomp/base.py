from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, NamedTuple, TypeVar, get_args, get_origin

Step = Any  # NOTE - don't import the real class, as it causes circular imports

T = TypeVar("T", contravariant=True)
RunState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]


def tostr(name, x, level: int = 0):
    prefix = "  " * level
    prefixln = "\n" + prefix
    return prefix + name + "=[" + prefixln + "  " + (",\n  " + prefix).join([*map(str, x)]) + prefixln + "]"


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
            print(f"{prefix}->[Model] {cls.__name__}")
            print(f"{prefix}  parameters [w/default] ({len(op)}):", [*map(str, op)])
            print(f"{prefix}  parameters [required] ({len(rp)}):", [*map(str, rp)])
            print(f"{prefix}  observables ({len(ob)}):", [*map(str, ob)])
            print()
        return {*rp, *op}, ob

    def update(
        self,
        params: set[Parameter],
    ) -> None:
        """
        Update the model with the given parameters

        """
        for p in params:
            setattr(self, p.__class__.__name__, p)


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
            print(f"{prefix}->[Engine] {cls.__name__}")
            print(f"{prefix}  hyperparameters ({len(hp)}):", hp)
            print()
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
# Parameter Definitions
# --------------------------------------------


class Parameter(Generic[T]):
    """
    Base class for a parameter

    """

    initialized: bool = False
    typing: TypeVar
    value: T

    def __init__(self, t: TypeVar) -> None:
        super().__init__()
        self.typing = t

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


# --------------------------------------------
# Hyperparameter Definitions
# --------------------------------------------


class Hyperparameter(Generic[T]):
    """
    Base class for a hyperparameter

    """

    initialized = False
    typing: TypeVar
    value: T

    def __init__(self, t: TypeVar) -> None:
        super().__init__()
        self.typing = t

    def __call__(self, v: T) -> Hyperparameter:
        self.value = v
        self.initialized = True
        return self

    def __str__(self) -> str:
        if self.initialized:
            return f"H[{self.typing}]({self.value})"
        else:
            return f"H[{self.typing}](not set)"


# --------------------------------------------
# Observable Definitions
# --------------------------------------------


class Observable(Generic[T]):
    """
    Base class for an observation

    """

    initialized = False
    typing: TypeVar
    value: T

    def __init__(self, t: TypeVar) -> None:
        super().__init__()
        self.typing = t

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


class RunConfig(NamedTuple):
    params: Params
    hparams: Hparams
    obs: Obs


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
    def run(self, step: Step) -> RunState:
        """
        Run a computational step

        @param step: computational step to run
        @return: the execution state

        """

    @abstractmethod
    def poll(self, step: Step) -> RunState:
        """
        Poll the execution state of a computational step

        @param step: computational step to run
        @return: the execution state

        """

    @abstractmethod
    def fetch(self, step: Step, query: ObsQuery) -> ObsMap:
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
    def prepare(self, args: Args, level: int = 0) -> None:
        """
        Prepare to run for the given arguments

        @param args: the argument sets to prepare

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
