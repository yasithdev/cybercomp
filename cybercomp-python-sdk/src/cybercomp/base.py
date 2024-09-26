from __future__ import annotations

from abc import ABC
from typing import Generic, Sequence, TypeVar, get_args

import numpy as np

T = TypeVar("T", contravariant=True)
N = TypeVar("N", int, float)


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

    def run(self, runtime: Runtime) -> bool:
        """
        Execute on a given runtime to generate ouptuts

        """
        ...

    def fetch(self, runtime: Runtime, *observations: Observation) -> dict[Observation, Output]:
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
    required: bool
    typing: TypeVar

    def with_value(self, value: T) -> Parameter[T]:
        self.value = value
        return self

    def with_typing(self, tv: TypeVar) -> Parameter[T]:
        self.typing = tv
        return self

    @classmethod
    def Range(cls, start: N, end: N, num: int) -> Sequence[Parameter[N]]:
        space = np.linspace(start, end, num)
        params = list[Parameter[N]]()
        typ = get_args(cls)[0]
        for value in space:
            p = Parameter[typ]().with_value(value).with_typing(typ)
            params.append(p)
        return params


class Hyperparameter(Generic[T]):
    """
    Base class for a hyperparameter

    """

    value: T
    typing: TypeVar

    def with_value(self, value: T) -> Hyperparameter[T]:
        self.value = value
        return self

    def with_typing(self, tv: TypeVar) -> Hyperparameter[T]:
        self.typing = tv
        return self


class Observation(Generic[T]):
    """
    Base class for an observation

    """

    value: T
    typing: TypeVar

    def with_value(self, value: T) -> Observation[T]:
        self.value = value
        return self

    def with_typing(self, tv: TypeVar) -> Observation[T]:
        self.typing = tv
        return self


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
