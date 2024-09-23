from __future__ import annotations

from typing import Any, TypeVar, get_args

from .generics import Hyperparameter, Observation, Parameter, T

# types for a group of parameters, hyperparameters, observations, and outputs
Parameters = dict[type[Parameter[T]], T]
Hyperparameters = dict[type[Hyperparameter[T]], T]
Observations = dict[type[Observation[T]], T]
Output = Any


class TypeMismatchError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Model:
    """
    Base class for a Computational Model

    """

    def create_parameters(self, parameters: Parameters) -> list[Parameter]:
        P = list[Parameter]()
        for klass, value in parameters.items():
            tv: TypeVar = get_args(klass)[0]
            if tv.__name__ not in self.__dir__():
                raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{self.__class__.__name__}'")
                # ensure this parameter exists
            p = klass().with_typing(tv).with_value(value)
            P.append(p)
        return P

    def create_observations(self, observations: Observations) -> list[Observation]:
        O = list[Observation]()
        for klass, value in observations.items():
            tv: TypeVar = get_args(klass)[0]
            if tv.__name__ not in self.__dir__():
                raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{self.__class__.__name__}'")
                # ensure this observation exists
            o = klass().with_typing(tv).with_value(value)
            O.append(o)
        return O


class Engine:
    """
    Base class for a Computational Engine

    """

    def create_hyperparameters(self, hyperparameters: Hyperparameters) -> list[Hyperparameter]:
        H = list[Hyperparameter]()
        for klass, value in hyperparameters.items():
            tv: TypeVar = get_args(klass)[0]
            if tv.__name__ not in self.__dir__():
                raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{self.__class__.__name__}'")
                # ensure this hyperparameter exists
            h = klass().with_typing(tv).with_value(value)
            H.append(h)
        return H


class Runtime:
    """
    Base class for an execution context

    """

    pass
