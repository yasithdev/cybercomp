from typing import Any, Mapping, TypeVar

from cybercomp.base import ArgSet, Engine, Model, ObsSet

from .base import Runtime, tostr


class LocalRuntime(Runtime):

    def __init__(self) -> None:
        super().__init__()

    def run(self, model: Model, engine: Engine, argset: ArgSet, obsset: ObsSet) -> bool:
        # TODO correctly implement this
        print(f"LocalRuntime.run(), model={model}, engine={engine}", tostr("argset", argset), tostr("obsset", obsset))
        return True

    def fetch(self, model, engine, argset, obsset) -> Mapping[TypeVar, Any]:
        # TODO correctly implement this
        print(f"LocalRuntime.fetch(), model={model}, engine={engine}", tostr("argset", argset), tostr("obsset", obsset))
        return {}
