from typing import Any, TypeVar

from cybercomp.base import Engine, Model, ObsMap, ObsQuery, RunConfig
from cybercomp.experiment import Step

from .base import RunState, Runtime, tostr


class LocalRuntime(Runtime):

    def __init__(
        self,
        reuse_past_runs: bool = False,
    ) -> None:
        super().__init__()
        self.reuse_past_runs = reuse_past_runs

    def __enter__(self):
        print(f"Acquiring resource: {self.__class__.__name__}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Releasing resource: {self.__class__.__name__}")

    def run(self, step: Step) -> RunState:
        # TODO correctly implement this
        print(f"LocalRuntime.run()")
        print(f"model={step.model}")
        print(f"engine={step.engine}")
        assert step.config is not None
        step.describe()
        return "QUEUED"

    def poll(self, step: Step) -> RunState:
        # TODO correctly implement this
        print(f"LocalRuntime.poll()")
        print(f"model={step.model}")
        print(f"engine={step.engine}")
        assert step.config is not None
        step.describe()
        return "COMPLETED"

    def fetch(self, step: Step, query: ObsQuery) -> ObsMap:
        # TODO correctly implement this
        print(f"LocalRuntime.poll()")
        print(f"model={step.model}")
        print(f"engine={step.engine}")
        assert step.config is not None
        step.describe()
        print(tostr("query=", query))
        return {}
