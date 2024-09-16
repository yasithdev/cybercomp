from __future__ import annotations

from .base import Engine, Model, Runtime, Parameter, Hyperparameter, Observation
from .generics import T


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
    parameters: list[Parameter]
    hyperparameters: list[Hyperparameter]
    observations: list[Observation]
    runtime: Runtime

    def __init__(
        self,
        name: str,
        model: type[Model],
        engine: type[Engine],
        parameters: list[Parameter],
        hyperparameters: list[Hyperparameter],
        observations: list[Observation],
    ) -> None:
        self.name = name

        # convert parameters to dict
        _parameter_dict = {type(v).__str__(v): v for v in parameters}
        _observation_dict = {type(v).__str__(v): v for v in observations}
        _hyperparameter_dict = {type(v).__str__(v): v for v in hyperparameters}

        self.model = model(**_parameter_dict).with_observations(**_observation_dict)
        self.engine = engine(**_hyperparameter_dict)
        print(f"[Experiment] {name} is created")

    def execute(self, ctx: Runtime) -> bool:
        """
        Execute the experiment on a given runtime

        """
        print("run executed")
        return True

    def gather_observations(self, ctx: Runtime) -> dict[str, Observation]:
        """
        Gather generated observations from the runtime for post-analysis

        """
        print("observations gathered")
        return {}

    def __or__(self, expt: Experiment) -> Experiment:
        # this operator chains the *self* experiment with the *expt* experiment,
        # and creates a new experiment. In this operation, both parameters and
        # observations of the *self* experiment are accessible to the *expt*

        # parameters is the union of self.observations and expt.parameters
        parameters = pipe(self.observations, expt.parameters)
        expt_observations = expt.observations
        expt_hyperparameters = expt.hyperparameters

        return self


class Collection:
    """
    convenience class for defining collections of experiments.
    experiments are optimally chained when defined within a collection.
    """

    experiments: list[Experiment | Collection]

    def __init__(
        self,
        name: str,
        experiments: list[Experiment | Collection],
        parameters: list[Parameter] = [],
        hyperparameters: list[Hyperparameter] = [],
        observations: list[Observation] = [],
    ) -> None:
        self.experiments = []
        for i, item in enumerate(experiments):
            if isinstance(item, Experiment):
                experiment = Experiment(
                    name=f"{name}_{i}",
                    model=type(item.model),
                    engine=type(item.engine),
                    parameters=union(item.parameters, parameters),
                    hyperparameters=union(item.hyperparameters, hyperparameters),
                    observations=union(item.observations, observations),
                )
                self.experiments.append(experiment)
            elif isinstance(item, Collection):
                self.experiments.append(item)
        print(f"[Collection] {name} is created")

    @staticmethod
    def from_experiment(
        name: str,
        experiment: Experiment,
    ) -> Collection:
        collection = Collection(name=name, experiments=[experiment])
        print(f"[Collection] {name} is created")
        return collection

    def execute(self, ctx: Runtime) -> bool:
        """
        Execute the experiment on a given runtime

        """
        print("run executed")
        return True

    def gather_observations(self, ctx: Runtime) -> dict[str, Observation]:
        """
        Gather generated observations from the runtime for post-analysis

        """
        print("observations gathered")
        return {}


class Analysis(Experiment, Collection):
    pass
