from itertools import product
from typing import Iterator, Set, TypeVar

from .base import Args, Engine, Hyperparameter, Model, Observable, ObsMap, ObsQuery, Parameter, RunState, Runtime, T
from .experiment import Experiment, Step

N = TypeVar("N", int, float)


# --------------------------------------------
# Instantiation Functions
# --------------------------------------------


def parameter(typ: TypeVar, value: T) -> Parameter[T]:
    """
    Create a parameter with a given type and value

    @param typ: the type of the parameter
    @param value: the value of the parameter

    """
    return Parameter[T](typ)(value)


def hyperparameter(typ: TypeVar, value: T) -> Hyperparameter[T]:
    """
    Create a hyperparameter with a given type and value

    @param typ: the type of the hyperparameter
    @param value: the value of the hyperparameter

    """
    return Hyperparameter[T](typ)(value)


def observable(typ: TypeVar, value: T) -> Observable[T]:
    """
    Create an observable with a given type and value

    @param typ: the type of the observable
    @param value: the value of the observable

    """
    return Observable[T](typ)(value)


# --------------------------------------------
# Composition Functions
# --------------------------------------------


def experiment(model: Model, engine: Engine, name: str) -> Experiment:
    """
    Create an experiment with a single step

    @param model: the model to use
    @param engine: the engine to use
    @param name: the name of the experiment
    @return: the experiment

    """
    return Experiment(name, Step(name, model, engine))


def run_async(experiment: Experiment, args: Args, runtime: Runtime) -> Iterator[RunState]:
    """
    Run experiment with the given args (non-blocking)

    @param experiment: the experiment
    @param args: the args
    @param runtime: the runtime

    @return: the current execution state

    """
    experiment.prepare(args)
    return experiment.run(runtime=runtime)


def run_sync(experiment: Experiment, args: Args, runtime: Runtime) -> RunState:
    """
    Run experiment with the given args (blocking)

    @param experiment: the experiment
    @param args: the args
    @param runtime: the runtime

    @return: the current execution state

    """
    experiment.prepare(args)
    return experiment.run_sync(runtime=runtime)


def poll(
    experiment: Experiment,
    runtime: Runtime,
) -> Iterator[RunState]:
    """
    Poll the execution state of the experiment

    @param experiment: the experiment to run
    @param runtime: the runtime to use
    @return: the current execution state

    """
    return experiment.poll(runtime=runtime)


def fetch(
    experiment: Experiment,
    runtime: Runtime,
    query: ObsQuery = None,
) -> ObsMap:
    """
    Query for observations from the given run sets

    @param experiment: the experiment to run
    @param run: the run sets to run
    @param runtime: the runtime to use
    @param query: the observables to fetch
    @return: the generated observations

    """
    return experiment.fetch(runtime=runtime, query=query)


# --------------------------------------------
# Range Functions
# --------------------------------------------

Z = TypeVar("Z", Parameter, Hyperparameter)


def space(klass: type[Z], typ: TypeVar, start: N, end: N, num: int) -> Iterator[Z]:
    """
    Generate a space of values for a given type

    @param klass: the class to instantiate
    @param typ: the type of the values
    @param start: the start of the range
    @param end: the end of the range
    @param num: the number of values to generate

    """
    import numpy as np

    space = np.linspace(start, end, num)
    for value in space:
        assert type(value) in typ.__constraints__
        p = klass(typ)(value.item())
        yield p


def walk(
    param_space: Iterator[Set[Parameter]],
    hparam_space: Iterator[Set[Hyperparameter]],
) -> Iterator[Args]:
    """
    Generate a set of arguments from a given parameter/hyperparameter space

    @param param_space: a sequence of parameter sets. all items have the same set of parameters, with different values.
    @param hparam_space: a sequence of hyperparameter sets. all items have the same set of hyperparameters, with different values.

    @return: the set of arguments

    """
    # cross product of parameter and hyperparameter spaces
    for parameters, hyperparameters in product(param_space, hparam_space):
        yield {*parameters, *hyperparameters}


__all__ = [
    "parameter",
    "hyperparameter",
    "observable",
    "experiment",
    "run_async",
    "run_sync",
    "poll",
    "fetch",
    "space",
    "walk",
]
