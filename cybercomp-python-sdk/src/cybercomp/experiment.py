from __future__ import annotations

from typing import Any, Sequence, TypeVar, get_args

from .base import Engine, Hyperparameter, Model, Observation, Parameter, Runnable, Runtime, T
from .exceptions import TypeMismatchError
from .util_composition import (
    intersect_observations,
    merge_hyperparameters,
    merge_parameters_and_observations,
    union_observations,
)

ModelArgs = type[Model]
EngineArgs = type[Engine]
UnitContext = tuple[Model, Engine]
CollectionContext = Sequence[Runnable]
ParameterArgs = dict[type[Parameter[T]], T]
HyperparameterArgs = dict[type[Hyperparameter[T]], T]
ObservationArgs = dict[type[Observation[T]], T]
Output = Any


class Experiment(Runnable):

    name: str
    context: UnitContext | CollectionContext
    P: Sequence[Parameter]  # parameters
    H: Sequence[Hyperparameter]  # hyperparameters
    O: Sequence[Observation]  # observations

    def __init__(
        self,
        name: str,
        context: UnitContext | CollectionContext,
        P: Sequence[Parameter],
        H: Sequence[Hyperparameter],
        O: Sequence[Observation],
    ) -> None:
        self.name = name
        self.context = context
        self.P, self.H, self.O = P, H, O

    def describe(self):
        print(f"[Experiment] {self.name} is created")
        p_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in self.P])
        print(f"[P] {p_str}")
        h_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in self.H])
        print(f"[H] {h_str}")
        o_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in self.O])
        print(f"[O] {o_str}")

    @staticmethod
    def Unit(
        name: str,
        model: ModelArgs,
        engine: EngineArgs,
        parameters: ParameterArgs,
        hyperparameters: HyperparameterArgs,
        observations: ObservationArgs,
    ) -> Experiment:
        """
        class to define an experiment unit with a model, engine, parameters,
        hyperparameters, and observations.
        """
        context = model(), engine()
        P = create_parameters(context, parameters)
        H = create_hyperparameters(context, hyperparameters)
        O = create_observations(context, observations)
        exp = Experiment(name, context, P, H, O)
        exp.describe()
        return exp

    def run(self, runtime: Runtime) -> bool:
        # TODO implement this
        raise NotImplementedError()

    def fetch(self, runtime: Runtime, *observations: Observation) -> dict[Observation, Output]:
        # TODO implement this
        raise NotImplementedError()

    @staticmethod
    def Sequence(name: str, *experiments: Experiment) -> Sequential:
        """
        Define a chain of experiments, where each experiment depends
        on outputs from the previous experiment.

        """
        return Sequential(name, *experiments)

    @staticmethod
    def Sweep(
        name: str,
        context: UnitContext | CollectionContext,
        parameter_space: set[ParameterArgs],
        hyperparameter_space: set[HyperparameterArgs],
        observations: ObservationArgs,
    ) -> Parallel:
        """
        Define an array of independent experiments that run on the given
        space of input parameters and hyperparameters (an experiment per
        item in that space). Each experiment can be run in parallel.
        Same observations will be passed to all experiments.

        """
        # cross product of parameter and hyperparameter spaces
        experiments = []
        for param in parameter_space:
            for hparam in hyperparameter_space:
                P = create_parameters(context, param)
                H = create_hyperparameters(context, hparam)
                O = create_observations(context, observations)
                exp = Experiment(name, context, P, H, O)
                experiments.append(exp)
        return Parallel(name, *experiments)


class Sequential(Experiment):
    """
    A collection of experiments that must run sequentially

    """

    def __init__(
        self,
        name: str,
        *experiments: Experiment,
    ) -> None:
        assert len(experiments) > 0
        context = list[Experiment]()
        for i, unit in enumerate(experiments):
            print(f"[Sequential][{i}] {unit.name}")
            if i == 0:
                context.append(unit)
            else:
                # carry P/H/O over to the next unit
                x_prev = context[-1]
                unit.P = merge_parameters_and_observations(x_prev.P, x_prev.O, unit.P)
                unit.H = merge_hyperparameters(x_prev.H, unit.H)
                unit.O = union_observations(x_prev.O, unit.O)  # can take observations from any unit
                context.append(unit)
        # set P/H/O
        last = context[-1]
        super().__init__(name, context, last.P, last.H, last.O)

    def run(self, runtime: Runtime) -> bool:
        raise NotImplementedError()

    def fetch(self, runtime: Runtime, *observations: Observation) -> dict[Observation, Output]:
        raise NotImplementedError()


class Parallel(Experiment):
    """
    A collection of experiments that can run in parallel

    """

    def __init__(
        self,
        name: str,
        *experiments: Experiment,
    ) -> None:
        context = experiments
        P = list[Parameter]()
        H = list[Hyperparameter]()
        O = list[Observation]()
        for i, unit in enumerate(experiments):
            print(f"[Parallel][{i}] {unit.name}")
            if i == 0:
                P, H, O = unit.P, unit.H, unit.O
            else:
                # take union of P/H/O
                P = merge_parameters_and_observations(P, unit.O, unit.P)
                H = merge_hyperparameters(H, unit.H)
                O = intersect_observations(O, unit.O)  # only common observations can be taken
        super().__init__(name, context, P, H, O)

    def run(self, runtime: Runtime) -> bool:
        raise NotImplementedError()

    def fetch(self, runtime: Runtime, *observations: Observation) -> dict[Observation, Output]:
        raise NotImplementedError()


def create_parameters(context: UnitContext | CollectionContext, parameters: ParameterArgs) -> Sequence[Parameter]:
    if len(context) == 2 and isinstance(context[0], Model) and isinstance(context[1], Engine):
        M: Model = context[0]
        names = M.__dir__()
        obj_name = M.__class__.__name__
    else:
        names = set()
        objs = []
        for exp in context:
            assert isinstance(exp, Experiment)
            names.update([p.typing.__name__ for p in exp.P])
            objs.append(exp.name)
        obj_name = "|".join(objs)

    P = list[Parameter]()
    for klass, value in parameters.items():
        tv: TypeVar = get_args(klass)[0]
        if tv.__name__ not in names:
            raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{obj_name}'")
            # ensure this parameter exists
        p = klass().with_typing(tv).with_value(value)
        P.append(p)
    return P


def create_hyperparameters(
    context: UnitContext | CollectionContext, hyperparameters: HyperparameterArgs
) -> Sequence[Hyperparameter]:
    if len(context) == 2 and isinstance(context[0], Model) and isinstance(context[1], Engine):
        E: Engine = context[1]
        names = E.__dir__()
        obj_name = E.__class__.__name__
    else:
        names = set()
        objs = []
        for exp in context:
            assert isinstance(exp, Experiment)
            names.update([h.typing.__name__ for h in exp.H])
            objs.append(exp.name)
        obj_name = "|".join(objs)
    H = list[Hyperparameter]()
    for klass, value in hyperparameters.items():
        tv: TypeVar = get_args(klass)[0]
        if tv.__name__ not in names:
            raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{obj_name}'")
            # ensure this hyperparameter exists
        h = klass().with_typing(tv).with_value(value)
        H.append(h)
    return H


def create_observations(
    context: UnitContext | CollectionContext, observations: ObservationArgs
) -> Sequence[Observation]:
    if len(context) == 2 and isinstance(context[0], Model) and isinstance(context[1], Engine):
        M: Model = context[0]
        names = M.__dir__()
        obj_name = M.__class__.__name__
    else:
        names = set()
        objs = []
        for exp in context:
            assert isinstance(exp, Experiment)
            names.update([o.typing.__name__ for o in exp.O])
            objs.append(exp.name)
        obj_name = "|".join(objs)

    O = list[Observation]()
    for klass, value in observations.items():
        tv: TypeVar = get_args(klass)[0]
        if tv.__name__ not in names:
            raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{obj_name}'")
            # ensure this observation exists
        o = klass().with_typing(tv).with_value(value)
        O.append(o)
    return O
