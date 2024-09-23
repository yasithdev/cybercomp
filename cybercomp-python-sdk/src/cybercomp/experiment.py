from __future__ import annotations

from .base import Engine, Model, Observation, Runtime, Parameters, Hyperparameters, Observations, Output
from .generics import Hyperparameter, Parameter, T


def union(a: list[T], b: list[T]) -> list[T]:
    return [*a, *b]


def intersection(A: list[T], B: list[T]) -> list[T]:
    inter: list[T] = []
    a_args: list[str] = [type(a).__str__(a) for a in A]
    b_args: list[str] = [type(b).__str__(b) for b in B]
    for i, a in enumerate(a_args):
        if a in b_args:
            inter.append(A[i])
    return inter


def pipe(a: list[Observation], b: list[Parameter]) -> list[Parameter]:
    return b


class Experiment:
    """
    class to define a concrete experiment.
    expects a concrete model and engine to initialize.
    parameters, hyperparameters, and observations are provided type-free for now

    """

    name: str
    model: Model
    engine: Engine
    runtime: Runtime
    P: list[Parameter]
    H: list[Hyperparameter]
    O: list[Observation]

    def __init__(
        self,
        name: str,
        model: type[Model],
        engine: type[Engine],
        parameters: Parameters,
        hyperparameters: Hyperparameters,
        observations: Observations,
    ) -> None:
        self.name = name
        self.model = model()
        self.engine = engine()
        self.P = self.model.create_parameters(parameters)
        self.H = self.engine.create_hyperparameters(hyperparameters)
        self.O = self.model.create_observations(observations)
        print(f"[Experiment] {name} is created")
        p_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in self.P])
        print(f"[P] {p_str}")
        h_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in self.H])
        print(f"[H] {h_str}")
        o_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in self.O])
        print(f"[O] {o_str}")

    def execute(self, ctx: Runtime) -> bool:
        """
        Execute the experiment on a given runtime

        """
        print("run executed")
        return True

    def retrieve(self, ctx: Runtime, *observations: Observation) -> dict[Observation, Output]:
        """
        Gather generated outputs from the runtime for post-analysis

        """
        print("outputs gathered", observations)
        return {}


class Collection:
    """
    convenience class for defining collections of experiments.
    experiments are optimally chained when defined within a collection.
    """

    name: str
    experiments: tuple[Experiment | Collection, ...]

    def __init__(
        self,
        name: str,
        *experiments: Experiment | Collection,
    ) -> None:
        self.name = name
        self.experiments = experiments
        print(f"[Collection] {name} is created")

    @staticmethod
    def From(
        name: str,
        experiment: Experiment,
    ) -> Collection:
        collection = Collection(name, experiment)
        print(f"[Collection] {name} is created")
        return collection
    
    @staticmethod
    def Sweep(
        name: str,
        model: type[Model],
        engine: type[Engine],
        runtime: Runtime,
        parameter_space: set[Parameters],
        hyperparameter_space: set[Hyperparameters],
        observations: Observations,
    ) -> Parallel:
        # cross product of parameter and hyperparameter spaces
        experiments = []
        for param in parameter_space:
            for hparam in hyperparameter_space:
                exp = Experiment(name, model, engine, param, hparam, observations)
                experiments.append(exp)
        return Parallel(name, *experiments)
    
    @staticmethod
    def Sequential(
        name: str,
        *experiments: Experiment | Collection
    ) -> Sequential:
        return Sequential(name, *experiments)

    def execute(self, ctx: Runtime) -> bool:
        """
        Execute the experiment on a given runtime

        """
        print("run executed")
        return True

    def retrieve(self, ctx: Runtime, *observations: Observation) -> dict[Observation, Output]:
        """
        Gather generated observations from the runtime for post-analysis

        """
        print("observations gathered", observations)
        return {}


class Parallel(Collection):
    """
    A collection of parallel experiments

    """

    def __init__(
        self,
        name: str,
        *experiments: Experiment | Collection,
    ) -> None:
        super().__init__(name, *experiments)


class Sequential(Collection):
    """
    A collection of sequential experiments

    """

    def __init__(
        self,
        name: str,
        *experiments: Experiment | Collection,
    ) -> None:
        name = "+".join([e.name for e in experiments])
        xList = list[Experiment|Collection]()
        for i, item in enumerate(experiments):
            if i == 0:
                # root element
                xList.append(item)
            else:
                x_prev = xList[-1]
                if isinstance(item, Experiment):
                    x_next = Experiment(
                        name=f"{name}_{i}",
                        model=type(item.model),
                        engine=type(item.engine),
                        parameters=union(x_prev.parameters, parameters),
                        hyperparameters=union(item.Harameters, hyperparameters),
                        observations=union(item.observations, observations),
                    )
                xList.append(experiment)
            elif isinstance(item, Collection):
                xList.append(item)
        super().__init__(name, *xList)

    def then(
        self,
        expt: Experiment | Collection,
    ) -> Collection:
        self.Nx = expt
        expt.Px = self
        return expt


class Analysis(Experiment, Collection):
    pass
