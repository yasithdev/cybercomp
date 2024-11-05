from typing import Any, TypeVar

from cybercomp.base import Engine, Model, ObsMap, ObsQuery, RunConfig
from cybercomp.experiment import Step
from chevron import render

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

    def run(self, step: Step, level: int = 0) -> RunState:
        # TODO correctly implement this
        prefix = "  " * level
        print(f"{prefix}->[LocalRuntime] run()")
        # generate the command to run - simple substitution for now
        assert step.config is not None
        data = dict(parameters={}, hyperparameters={}, observables={})
        for p in step.config.params:
            k, v = p.typing.__name__, p.value
            assert k not in data["parameters"]
            data["parameters"][k] = v
        for h in step.config.hparams:
            k, v = h.typing.__name__, h.value
            assert k not in data["hyperparameters"]
            data["hyperparameters"][k] = v
        for o in step.config.obs:
            k, v = o.typing.__name__, o.value
            assert k not in data["observables"]
            data["observables"][k] = v
        command = render(step.model.run_command, data).strip()
        print(f"{prefix}    model=  {step.model.__class__.__name__}")
        print(f"{prefix}    engine= {step.engine.__class__.__name__}")
        print(f"{prefix}    command={command}")
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
