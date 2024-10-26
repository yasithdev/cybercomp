from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, NamedTuple, TypeVar, get_args, get_origin

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
    def describe(cls, level: int = 0) -> tuple[set[RequiredParameter], set[DefaultParameter], set[Observable]]:
        """
        Describe the model

        """
        rp, op, ob = set(), set(), set()
        trp = set[RequiredParameter]()
        top = set[DefaultParameter]()
        tob = set[Observable]()
        for k, v in cls.__dict__.items():
            origin = typ = None
            if hasattr(v, "typing"):
                typ = v.typing
            if hasattr(v, "__class__"):
                origin = v.__class__
            if origin is None or typ is None:
                continue
            if issubclass(origin, RequiredParameter):
                rp.add(k)
                trp.add(typ)
            if issubclass(origin, DefaultParameter):
                op.add(k)
                top.add(typ)
            if issubclass(origin, Observable):
                ob.add(k)
                tob.add(typ)
        prefix = "  " * level
        print(f"{prefix}->[Model] {cls.__name__}")
        print(f"{prefix}  parameters [w/default] ({len(op)}):", op)
        print(f"{prefix}  parameters [required] ({len(rp)}):", rp)
        print(f"{prefix}  observables ({len(ob)}):", ob)
        print()
        return (trp, top, tob)


class Engine:
    """
    Base class for a Computational Engine

    """

    @classmethod
    def describe(cls, level: int = 0) -> set[Hyperparameter]:
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
                thp.add(typ[0])
        prefix = "  " * level
        print(f"{prefix}->[Engine] {cls.__name__}")
        print(f"{prefix}  hyperparameters ({len(hp)}):", hp)
        print()
        return thp


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

ArgSet = set[Parameter | Hyperparameter]
ObsSet = set[Observable]
RunSet = NamedTuple("RunSet", [("args", ArgSet), ("obs", ObsSet)])

Observation = Any
OutSet = set[Observation]

ObsQuery = set[Observable] | None
ObsMap = dict[Observable, Observation]
ObservationSet = set[set[Observable]]


class Runtime(ABC):
    """
    Base class for an execution context

    """

    @abstractmethod
    def run(self, model: Model, engine: Engine, run: RunSet) -> RunState:
        """
        Start a computational run using the given information

        @param model: the model to run
        @param engine: the engine to run
        @param run: the run identifier
        @return: the run status

        """

    @abstractmethod
    def poll(self, model: Model, engine: Engine, run: RunSet) -> RunState:
        """
        Poll the execution status of the computational run

        @param model: the model to poll
        @param engine: the engine to poll
        @param run: the run identifier
        @return: the run status

        """

    @abstractmethod
    def fetch(self, model: Model, engine: Engine, run: RunSet, query: ObsQuery) -> ObsMap:
        """
        Fetch observations from a completed computational run

        @param model: the model to fetch
        @param engine: the engine to fetch
        @argset: the argument set (identifier)
        @obsset: the observable set (identifier)
        @return: the fetched observations

        """


class Runnable(ABC):
    """
    Base class for a runnable object

    """

    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def describe(self, level: int = 0) -> None:
        """
        Describe the runnable object

        """

    @abstractmethod
    def prepare(self, args: ArgSet, level: int = 0) -> RunSet:
        """
        Generate run set for the given arg set

        @param args: the argument sets to prepare
        @return: the run sets

        """

    @abstractmethod
    def run(self, run: RunSet, runtime: Runtime, level: int = 0) -> RunState:
        """
        Execute the run set on a runtime

        @param args: the run sets to run
        @return: the run statuses

        """

    @abstractmethod
    def fetch(self, run: RunSet, runtime: Runtime, observables: ObsQuery = None, level: int = 0) -> ObsMap:
        """
        Fetch observations generated by the run set from the runtime

        @param args: the run sets to run
        @return: the fetched observables

        """
