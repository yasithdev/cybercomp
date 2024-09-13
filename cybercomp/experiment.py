from __future__ import annotations

from typing import Any

from .base import Engine, Model, Runtime


class Experiment:
    """
    class to define a concrete experiment.
    expects a concrete model and engine to initialize.
    parameters, hyperparameters, and observations are provided type-free for now

    """

    name: str
    model: Model
    engine: Engine
    parameters: dict[str, str]
    hyperparameters: dict[str, str]
    observations: dict[str, str]
    runtime: Runtime

    def __init__(
        self,
        name: str,
        model: type[Model],
        engine: type[Engine],
        parameters: dict[str, str],
        hyperparameters: dict[str, str],
        observations: dict[str, str],
    ) -> None:
        self.name = name
        self.model = model(**parameters)
        self.engine = engine(**hyperparameters)
        self.parameters = parameters
        self.hyperparameters = hyperparameters
        self.observations = observations

    def execute(self, ctx: Runtime) -> bool:
        """
        Execute the experiment on a given runtime

        """
        print("run executed")
        return True

    def gather_observations(self, ctx: Runtime) -> dict[str, Any]:
        """
        Gather generated observations from the runtime for post-analysis

        """
        print("observations gathered")
        return {}


class Collection:
    """
    convenience class for defining collections of experiments.
    experiments are optimally chained when defined within a collection.
    """

    experiments: list[Experiment | Collection]

    def __init__(
        self,
        name: str,
        models: list[type[Model] | Collection],
        engine: type[Engine],
        parameters: dict[str, str],
        hyperparameters: dict[str, str],
        observations: dict[str, str],
    ) -> None:
        self.experiments = []
        for i, item in enumerate(models):
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

    def execute(self, ctx: Runtime) -> bool:
        """
        Execute the experiment on a given runtime

        """
        print("run executed")
        return True

    def gather_observations(self, ctx: Runtime) -> dict[str, Any]:
        """
        Gather generated observations from the runtime for post-analysis

        """
        print("observations gathered")
        return {}


class Analysis(Experiment, Collection):
    pass
