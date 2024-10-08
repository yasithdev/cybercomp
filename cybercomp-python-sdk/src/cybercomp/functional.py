from typing import Sequence, TypeVar, get_args

from .base import Engine, Hyperparameter, Model, Observable, Parameter, T
from .experiment import Experiment, Step

N = TypeVar("N", int, float)


# --------------------------------------------
# Instantiation Functions
# --------------------------------------------


def parameter(typ: TypeVar, value: T) -> Parameter[T]:
    return Parameter[T](typ, value)


def hyperparameter(typ: TypeVar, value: T) -> Hyperparameter[T]:
    return Hyperparameter[T](typ, value)


def observable(typ: TypeVar, value: T) -> Observable[T]:
    return Observable[T](typ, value)


# --------------------------------------------
# Range Functions
# --------------------------------------------


def parameter_range(klass: type[Parameter], start: N, end: N, num: int) -> Sequence[Parameter[N]]:
    import numpy as np

    space = np.linspace(start, end, num)
    P = list[Parameter[N]]()
    typ = get_args(klass)[0]
    for value in space:
        p = Parameter[typ](typ, value.item())
        P.append(p)
    return P


def hyperparameter_range(klass: type[Hyperparameter], start: N, end: N, num: int) -> Sequence[Hyperparameter[N]]:
    import numpy as np

    space = np.linspace(start, end, num)
    H = list[Hyperparameter[N]]()
    typ = get_args(klass)[0]
    for value in space:
        p = Hyperparameter[typ](typ, value.item())
        H.append(p)
    return H


# --------------------------------------------
# Composition Functions
# --------------------------------------------


def experiment(*steps: Experiment, name: str | None = None) -> Experiment:
    if name is None:
        name = "+".join([s.name for s in steps])
    return Experiment(name, *steps)


def step(model: Model, engine: Engine, name: str) -> Experiment:
    return Experiment(name, Step(name, model, engine))


__all__ = ["parameter", "hyperparameter", "observable", "parameter_range", "hyperparameter_range", "experiment", "step"]
