from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Sequence, TypeVar, get_origin, get_args, Literal

T = TypeVar("T", contravariant=True)
RunState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]


def tostr(name, x, level: int = 0):
    prefix = "  " * level
    prefixln = "\n" + prefix
    return prefix + name + "=[" + prefixln + "  " + (",\n  "+prefix).join([*map(str, x)]) + prefixln + "]"


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
    def describe(cls, level: int = 0) -> tuple[Sequence[RequiredParameter], Sequence[OptionalParameter], Sequence[TypeVar]]:
        """
        Describe the model

        """
        rp, op, ob = [], [], []
        trp = list[RequiredParameter]()
        top = list[OptionalParameter]()
        tob = list[TypeVar]()
        for k, v in cls.__dict__.items():
            origin, typ = get_origin(v), get_args(v)
            if origin is None:
                continue
            if issubclass(origin, RequiredParameter):
                rp.append(k)
                trp.append(typ[0])
            if issubclass(origin, OptionalParameter):
                op.append(k)
                top.append(typ[0])
            if issubclass(origin, Observable):
                ob.append(k)
                tob.append(typ[0])
        prefix = "  " * level
        print(f"{prefix}->[Model] {cls.__name__}")
        print(f"{prefix}  required parameters ({len(rp)}):", rp)
        print(f"{prefix}  optional Parameters ({len(op)}):", op)
        print(f"{prefix}  observables ({len(ob)}):", ob)
        print()
        return (trp, top, tob)


class Engine:
    """
    Base class for a Computational Engine

    """

    @classmethod
    def describe(cls, level: int = 0) -> Sequence[Hyperparameter]:
        """
        Describe the engine

        """
        hp = []
        thp = list[Hyperparameter]()
        for k, v in cls.__dict__.items():
            origin, typ = get_origin(v), get_args(v)
            if origin is None:
                continue
            if issubclass(origin, Hyperparameter):
                hp.append(k)
                thp.append(typ[0])
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

    value: T
    typing: TypeVar
    required: bool

    def __init__(self, t: TypeVar, v: T) -> None:
        super().__init__()
        self.value = v
        self.typing = t

    def __str__(self) -> str:
        return f"P[{self.typing}]({self.value})"


class RequiredParameter(Parameter[T]):
    """
    Base class for a required parameter

    """

    required = True


class OptionalParameter(Parameter[T]):
    """
    Base class for an optional parameter

    """

    required = False


# --------------------------------------------
# Hyperparameter Definitions
# --------------------------------------------


class Hyperparameter(Generic[T]):
    """
    Base class for a hyperparameter

    """

    value: T
    typing: TypeVar

    def __init__(self, t: TypeVar, v: T) -> None:
        super().__init__()
        self.value = v
        self.typing = t

    def __str__(self) -> str:
        return f"H[{self.typing}]({self.value})"


# --------------------------------------------
# Observable Definitions
# --------------------------------------------


class Observable(Generic[T]):
    """
    Base class for an observation

    """

    value: T
    typing: TypeVar

    def __init__(self, t: TypeVar, v: T) -> None:
        super().__init__()
        self.value = v
        self.typing = t

    def __str__(self) -> str:
        return f"O[{self.typing}]({self.value})"


# --------------------------------------------
# Execution Types and Definitions
# --------------------------------------------

ArgSet = Sequence[Parameter | Hyperparameter]
ObsSet = Sequence[TypeVar]
RunSet = tuple[ArgSet, ObsSet]

Observation = Any
OutSet = Sequence[Observation]

ObsQuery = Sequence[TypeVar] | None
ObsMap = Mapping[TypeVar, Observation]
ObservationSet = Sequence[Sequence[Observable]]


class Runtime(ABC):
    """
    Base class for an execution context

    """

    @abstractmethod
    def run(self, model, engine, argset, obsset) -> RunState:
        """
        Execute the runtime

        @param model: the model to run
        @param engine: the engine to run
        @argset: the argument set to run
        @obsset: the observable set to run
        @return: the run status

        """

    @abstractmethod
    def poll(self, model, engine, argset, obsset) -> RunState:
        """
        Poll the runtime for the run status

        @param model: the model to poll
        @param engine: the engine to poll
        @argset: the argument set to poll
        @obsset: the observable set to poll
        @return: the run status

        """

    @abstractmethod
    def fetch(self, model, engine, argset, obsset) -> ObsMap:
        """
        Fetch the runtime

        @param model: the model to fetch
        @param engine: the engine to fetch
        @argset: the argument set to fetch
        @obsset: the observable set to fetch
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
    def prepare(self, *args: ArgSet, level: int = 0) -> Sequence[RunSet]:
        """
        Generate run sets for the given arg sets

        @param args: the argument sets to prepare
        @return: the run sets

        """

    @abstractmethod
    def run(self, *args: RunSet, runtime: Runtime, level: int = 0) -> Sequence[RunState]:
        """
        Execute the run sets on a runtime

        @param args: the run sets to run
        @return: the run statuses

        """

    @abstractmethod
    def fetch(
        self, *args: RunSet, runtime: Runtime, observables: ObsQuery = None, level: int = 0
    ) -> Sequence[ObsMap]:
        """
        Fetch observations generated by the run sets from the runtime

        @param args: the run sets to run
        @return: the fetched observables

        """
