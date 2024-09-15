from __future__ import annotations

from typing import Any

from .base import Engine, Model, Runtime, Parameter, Hyperparameter, Observation


class Experiment:
    """
    class to define a concrete experiment.
    expects a concrete model and engine to initialize.
    parameters, hyperparameters, and observations are provided type-free for now

    """

    name: str
    model: Model
    engine: Engine
    parameters: dict[str, Parameter]
    hyperparameters: dict[str, Hyperparameter]
    observations: dict[str, Observation]
    runtime: Runtime

    def __init__(
        self,
        name: str,
        model: type[Model],
        engine: type[Engine],
        parameters: dict[str, Parameter],
        hyperparameters: dict[str, Hyperparameter],
        observations: dict[str, Observation],
    ) -> None:
        self.name = name
        self.model = model(**parameters).with_observations(**observations)
        self.engine = engine(**hyperparameters)
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

    def __or__(self, value: Any) -> Experiment:
        raise


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
        engine: type[Engine],
        parameters: dict[str, Parameter],
        hyperparameters: dict[str, Hyperparameter],
        observations: dict[str, Observation],
    ) -> None:
        self.experiments = []
        for i, item in enumerate(experiments):
            if isinstance(item, type):
                experiment = Experiment(
                    name=f"{name}_{i}",
                    model=item,
                    engine=engine,
                    parameters=parameters,
                    hyperparameters=hyperparameters,
                    observations=observations,
                )
                self.experiments.append(experiment)
            elif isinstance(item, Collection):
                self.experiments.append(item)
        print(f"[Collection] {name} is created")

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
