from itertools import product
from typing import Sequence, Set, TypeVar

from .base import (
    ArgSet,
    Engine,
    Hyperparameter,
    Model,
    Observable,
    ObsMap,
    ObsQuery,
    Parameter,
    RunSet,
    RunState,
    Runtime,
    T,
)
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
    return Parameter[T](typ, value)


def hyperparameter(typ: TypeVar, value: T) -> Hyperparameter[T]:
    """
    Create a hyperparameter with a given type and value

    @param typ: the type of the hyperparameter
    @param value: the value of the hyperparameter

    """
    return Hyperparameter[T](typ, value)


def observable(typ: TypeVar, value: T) -> Observable[T]:
    """
    Create an observable with a given type and value

    @param typ: the type of the observable
    @param value: the value of the observable

    """
    return Observable[T](typ, value)


# --------------------------------------------
# Range Functions
# --------------------------------------------

Z = TypeVar("Z", Parameter, Hyperparameter)


def vrange(klass: type[Z], typ: TypeVar, start: N, end: N, num: int) -> Sequence[Z]:
    """
    Generate a range of values for a given type

    @param klass: the class to instantiate
    @param typ: the type of the values
    @param start: the start of the range
    @param end: the end of the range
    @param num: the number of values to generate

    """
    import numpy as np

    space = np.linspace(start, end, num)
    P = list[Z]()
    for value in space:
        assert type(value) in typ.__constraints__
        p = klass(typ, value.item())
        P.append(p)
    return P


def args(
    parameter_space: Sequence[Set[Parameter]],
    hyperparameter_space: Sequence[Set[Hyperparameter]],
) -> Sequence[ArgSet]:
    """
    Generate a set of arguments from a given parameter/hyperparameter space

    @param parameter_space: a sequence of parameter sets.
                            all items have the same set of parameters,
                            but with different values.
    @hyperparameter_space:  a sequence of hyperparameter sets.
                            all items have the same set of hyperparameters,
                            but with different values.

    """
    # cross product of parameter and hyperparameter spaces
    argsets = list[ArgSet]()
    for parameters, hyperparameters in product(parameter_space, hyperparameter_space):
        argset = [*parameters, *hyperparameters]
        argsets.append(argset)
    return argsets


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


def prepare(
    experiment: Experiment,
    *args: ArgSet,
) -> Sequence[RunSet]:
    """
    Prepare the experiment by generating run sets for the given arg sets

    @param experiment: the experiment to prepare
    @param args: the argument sets to prepare
    @return: the run sets

    """
    return experiment.prepare(*args)

def begin_run(
    experiment: Experiment,
    *args: RunSet,
    runtime: Runtime,
) -> Sequence[Sequence[RunState]]:
    """
    Begin running the experiment with the given run sets

    @param experiment: the experiment to run
    @param args: the run sets to run
    @param runtime: the runtime to use
    @return: the run statuses

    """
    return experiment.run(*args, runtime=runtime)


def wait_for_completion(
    experiment: Experiment,
    *args: RunSet,
    runtime: Runtime,
    poll_every_n_secs: int = 2,
) -> bool:
    """
    Block until experiment reaches a final state

    @param experiment: the experiment to run
    @param args: the run sets to run
    @param runtime: the runtime to use
    @return: the run statuses

    """
    import time

    while True:
        done = []
        runsets_state = experiment.poll(*args, runtime=runtime)
        for runset_state in runsets_state:
            is_done = all(state in ["COMPLETED", "FAILED"] for state in runset_state)
            done.append(is_done)
        if all(done):
            return True
        else:
            time.sleep(poll_every_n_secs)

def poll(
    experiment: Experiment,
    *args: RunSet,
    runtime: Runtime,
) -> Sequence[Sequence[RunState]]:
    """
    Poll the experiment with the given run sets

    @param experiment: the experiment to run
    @param args: the run sets to run
    @param runtime: the runtime to use
    @return: the run statuses

    """
    return experiment.poll(*args, runtime=runtime)

def run(
    experiment: Experiment,
    *args: ArgSet,
    runtime: Runtime,
):
    """
    Run the experiment with the given run sets and wait for completion

    @param experiment: the experiment to run
    @param args: the run sets to run
    @param runtime: the runtime to use
    @return: the run statuses

    """
    runsets = prepare(experiment, *args)
    begin_run(experiment, *runsets, runtime=runtime)
    wait_for_completion(experiment, *runsets, runtime=runtime)
    return experiment, *runsets

def fetch(
    experiment: Experiment,
    *args: RunSet,
    runtime: Runtime,
    observables: ObsQuery = None,
) -> Sequence[ObsMap]:
    """
    Fetch the observables for the given run sets

    @param experiment: the experiment to run
    @param args: the run sets to run
    @param runtime: the runtime to use
    @param observables: the observables to fetch
    @return: the fetched observables

    """
    return experiment.fetch(*args, runtime=runtime, observables=observables)


__all__ = ["parameter", "hyperparameter", "observable", "vrange", "prepare", "run", "fetch", "experiment"]
