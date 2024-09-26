from __future__ import annotations

from abc import ABC
from typing import Sequence

from .base import Engine, HyperparameterArgs, Model, Observation, ObservationArgs, Output, ParameterArgs, Runtime
from .generics import Hyperparameter, Parameter

ModelArgs = type[Model]
EngineArgs = type[Engine]


def observation_to_parameter(o: Observation) -> Parameter:
    # TODO implement this
    raise


def merge_parameters_and_observations(
    P1: Sequence[Parameter], O1: Sequence[Observation], P2: Sequence[Parameter]
) -> Sequence[Parameter]:
    out: Sequence[Parameter] = []
    for p in P1:
        out.append(p)
    for o in O1:
        out.append(observation_to_parameter(o))
    for p in P2:
        # TODO properly implement
        if p in out:
            # when parameters are duplicated, enforce priority for newer one
            out.pop(out.index(p))
        out.append(p)
    return out


def merge_hyperparameters(H1: Sequence[Hyperparameter], H2: Sequence[Hyperparameter]) -> Sequence[Hyperparameter]:
    return [*H1, *H2]


def union_observations(O1: Sequence[Observation], O2: Sequence[Observation]) -> Sequence[Observation]:
    return [*O1, *O2]


def intersect_observations(O1: Sequence[Observation], O2: Sequence[Observation]) -> Sequence[Observation]:
    out: Sequence[Observation] = []
    for o in O2:
        # TODO properly implement
        if o in O1:
            out.append(o)
    return out


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


class Experiment(Runnable):
    name: str
    context: tuple[Model, Engine] | Sequence[Experiment]

    P: Sequence[Parameter]  # parameters
    H: Sequence[Hyperparameter]  # hyperparameters
    O: Sequence[Observation]  # observations

    def __init__(self, name: str) -> None:
        self.name = name

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
        exp = Experiment(name)
        M, E = model(), engine()
        exp.context = (M, E)
        # generate P/H/O based on the given types and context
        exp.P = M.create_parameters(parameters)
        exp.H = E.create_hyperparameters(hyperparameters)
        exp.O = M.create_observations(observations)
        print(f"[Experiment] {name} is created")
        p_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in exp.P])
        print(f"[P] {p_str}")
        h_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in exp.H])
        print(f"[H] {h_str}")
        o_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in exp.O])
        print(f"[O] {o_str}")
        return exp

    @staticmethod
    def Chain(name: str, *experiments: Experiment) -> Sequential:
        """
        Define a chain of experiments, where each experiment depends
        on outputs from the previous experiment.

        """
        return Sequential(name, *experiments)

    @staticmethod
    def Sweep(
        name: str,
        model: type[Model],
        engine: type[Engine],
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
                exp = Experiment.Unit(name, model, engine, param, hparam, observations)
                experiments.append(exp)
        return Parallel(name, *experiments)

    def run(self, runtime: Runtime) -> bool:
        raise NotImplementedError()

    def fetch(self, runtime: Runtime, *observations: Observation) -> dict[Observation, Output]:
        raise NotImplementedError()


class Sequential(Experiment):
    """
    A collection of experiments that must run sequentially

    """

    def __init__(
        self,
        name: str,
        *experiments: Experiment,
    ) -> None:
        super().__init__(name)
        assert len(experiments) > 0
        self.context = list[Experiment]()
        for i, unit in enumerate(experiments):
            if i == 0:
                self.context.append(unit)
            else:
                print(f"[Sequential][{i}] {unit.name}")
                # carry P/H/O over to the next unit
                x_prev = self.context[-1]
                unit.P = merge_parameters_and_observations(x_prev.P, x_prev.O, unit.P)
                unit.H = merge_hyperparameters(x_prev.H, unit.H)
                unit.O = union_observations(x_prev.O, unit.O)  # can take observations from any unit
                self.context.append(unit)
        # set P/H/O
        last = self.context[-1]
        self.P, self.H, self.O = last.P, last.H, last.O

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
        super().__init__(name)
        self.context = experiments
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
        self.P, self.H, self.O = P, H, O

    def run(self, runtime: Runtime) -> bool:
        raise NotImplementedError()

    def fetch(self, runtime: Runtime, *observations: Observation) -> dict[Observation, Output]:
        raise NotImplementedError()


class Analysis(Experiment):
    pass
