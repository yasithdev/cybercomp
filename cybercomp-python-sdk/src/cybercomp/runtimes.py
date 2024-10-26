from typing import Any, TypeVar

from cybercomp.base import Engine, Model, ObsMap, ObsQuery, RunSet

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

    def run(self, model: Model, engine: Engine, run: RunSet) -> RunState:
        # TODO correctly implement this
        print(f"LocalRuntime.run()")
        print(f"model={model}")
        print(f"engine={engine}")
        print(tostr("args=", run.args))
        print(tostr("obs=", run.obs))
        return "QUEUED"

    def poll(self, model: Model, engine: Engine, run: RunSet) -> RunState:
        # TODO correctly implement this
        print(f"LocalRuntime.poll()")
        print(f"model={model}")
        print(f"engine={engine}")
        print(tostr("args=", run.args))
        print(tostr("obs=", run.obs))
        return "COMPLETED"

    def fetch(self, model: Model, engine: Engine, run: RunSet, query: ObsQuery) -> ObsMap:
        # TODO correctly implement this
        print(f"LocalRuntime.fetch()")
        print(f"model={model}")
        print(f"engine={engine}")
        print(tostr("args=", run.args))
        print(tostr("obs=", run.obs))
        print(tostr("query=", query))
        return {}
