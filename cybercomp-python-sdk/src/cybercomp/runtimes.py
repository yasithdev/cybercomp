from typing import Any, Mapping, TypeVar

from cybercomp.base import ArgSet, Engine, Model, ObsSet

from .base import Runtime, tostr, RunState


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

    def run(self, model: Model, engine: Engine, argset: ArgSet, obsset: ObsSet) -> RunState:
        # TODO correctly implement this
        print(f"LocalRuntime.run()")
        print(f"model={model}")
        print(f"engine={engine}")
        print(tostr("argset", argset))
        print(tostr("obsset", obsset))
        return "QUEUED"
    
    def poll(self, model: Model, engine: Engine, argset: ArgSet, obsset: ObsSet) -> RunState:
        # TODO correctly implement this
        print(f"LocalRuntime.poll()")
        print(f"model={model}")
        print(f"engine={engine}")
        print(tostr("argset", argset))
        print(tostr("obsset", obsset))
        return "COMPLETED"

    def fetch(self, model, engine, argset, obsset) -> Mapping[TypeVar, Any]:
        # TODO correctly implement this
        print(f"LocalRuntime.fetch()")
        print(f"model={model}")
        print(f"engine={engine}")
        print(tostr("argset", argset))
        print(tostr("obsset", obsset))
        return {}
