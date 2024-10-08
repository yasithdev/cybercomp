from __future__ import annotations

from abc import ABC
from typing import Any, Generic, Sequence, TypeVar, get_args

import numpy as np

T = TypeVar("T", contravariant=True)
N = TypeVar("N", int, float)
Observation = Any


class Model:
    """
    Base class for a Computational Model

    """


class Engine:
    """
    Base class for a Computational Engine

    """


class Runtime:
    """
    Base class for an execution context

    """


class Runnable(ABC):

    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(
        self,
        *args: Sequence[Parameter | Hyperparameter | Observable],
        runtime: Runtime,
    ) -> bool:
        """
        Execute on a given runtime to generate ouptuts

        """
        ...

    def fetch(
        self,
        *args: Sequence[Parameter | Hyperparameter | Observable],
        runtime: Runtime,
        observables: Sequence[Observable],
    ) -> Sequence[Sequence[Observation]]:
        """
        Gather generated outputs

        """
        ...


class Type(Generic[T]):
    """
    Base class for a semantic type

    """


class Category:
    """
    Base class for a category of semantic types

    """


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

    @staticmethod
    def Range(klass: type[Parameter], start: N, end: N, num: int) -> Sequence[Parameter[N]]:
        space = np.linspace(start, end, num)
        P = list[Parameter[N]]()
        typ = get_args(klass)[0]
        for value in space:
            p = Parameter[typ](typ, value.item())
            P.append(p)
        return P


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

    @staticmethod
    def Range(klass: type[Hyperparameter], start: N, end: N, num: int) -> Sequence[Hyperparameter[N]]:
        space = np.linspace(start, end, num)
        H = list[Hyperparameter[N]]()
        typ = get_args(klass)[0]
        for value in space:
            p = Hyperparameter[typ](typ, value.item())
            H.append(p)
        return H


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


def parameter(typ: TypeVar, value: T) -> Parameter[T]:
    return Parameter[T](typ, value)


def hyperparameter(typ: TypeVar, value: T) -> Hyperparameter[T]:
    return Hyperparameter[T](typ, value)


def observation(typ: TypeVar, value: T) -> Observable[T]:
    return Observable[T](typ, value)
