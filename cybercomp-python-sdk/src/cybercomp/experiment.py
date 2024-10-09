from __future__ import annotations

from typing import Sequence

from cybercomp.base import ArgSet, ObsMap, ObsQuery, RunSet

from .base import ArgSet, Engine, Model, Runnable, Runtime, tostr


class Step(Runnable):
    """
    A single computational step

    """

    model: Model
    engine: Engine

    def __init__(self, name: str, model: Model, engine: Engine) -> None:
        super().__init__(name)
        self.model = model
        self.engine = engine

    def prepare(self, *args: ArgSet) -> Sequence[RunSet]:
        # TODO correctly implement this
        out = []
        for argset in args:
            print(f"step=[{self.name}].prepare(), argset={tostr(argset)}")
            # first, ensure that argset satisfy model and engine requirements
            # next, concatenate the argset with model observables and return it
            (param_req, param_opt, observ) = self.model.describe()
            hparam = self.engine.describe()
            print(f"param_req={param_req}, param_opt={param_opt}, observ={observ}, hparam={hparam}")
            out.append(([*param_req, *param_opt], observ))
        return out

    def run(self, *args: RunSet, runtime: Runtime) -> Sequence[bool]:
        out = []
        for argset, obsset in args:
            print(f"step=[{self.name}].run(), argset={tostr(argset)}, obsset={tostr(obsset)}")
            status = runtime.run(self.model, self.engine, argset, obsset)
            out.append(status)
        return out

    def fetch(self, *args: RunSet, runtime: Runtime, **_) -> Sequence[ObsMap]:
        out = []
        for argset, obsset in args:
            print(f"step=[{self.name}].fetch(), argset={tostr(argset)}, obsset={tostr(obsset)}")
            obsmap = runtime.fetch(self.model, self.engine, argset, obsset)
            out.append(obsmap)
        return out
    
    def describe(self) -> None:
        print(f"[Step] {self.name}")
        print(f"* model={self.model.__class__.__name__}")
        print(f"* engine={self.engine.__class__.__name__}")
        print()
        self.model.describe()
        self.engine.describe()


class Experiment(Runnable):
    """
    A sequence of computational steps where
    each step depends on the previous step(s).

    """

    def __init__(self, name: str, *steps: Runnable) -> None:
        assert len(steps) > 0
        super().__init__(name)
        self.steps = steps

    def __rshift__(self, other: Experiment) -> Experiment:
        if not isinstance(other, Experiment):
            raise TypeError(f"Unsupported operand type(s) for >>: 'Experiment' and '{type(other).__name__}'")
        return Experiment(f"{self.name}+{other.name}", self, other)

    def __str__(self) -> str:
        return f"name=[{self.name}], steps=[" + ">>".join([s.name for s in self.steps]) + "]"

    def prepare(self, *args: ArgSet) -> Sequence[RunSet]:
        # TODO improve this
        from .util_composition import create_next_argset, create_next_obsset

        # do an independent run for each argset
        runsets = list[RunSet]()
        for argset in args:
            print(f"experiment=[{self.name}].prepare(), argset={tostr(argset)}")
            # no observations in the beginning
            obsset = []
            for step in self.steps:
                # step_argset is the subset of args that's used by step
                (step_argset, step_obsset) = step.prepare(argset)[0]
                next_argset = create_next_argset(argset, step_obsset)
                next_obsset = create_next_obsset(obsset, step_obsset)
                # update argset and obsset for the next step
                argset, obsset = next_argset, next_obsset
            # append final runset
            runsets.append((argset, obsset))

        return runsets
    
    def describe(self) -> None:
        print(f"[Experiment] {self.name}")
        print()
        for step in self.steps:
            step.describe()

    def run(self, *args: RunSet, runtime: Runtime) -> Sequence[bool]:
        # TODO verify this
        out = []
        for argset, obsset in args:
            print(f"experiment=[{self.name}].run(), argset={tostr(argset)}, obsset={tostr(obsset)}")
            status = True
            for step in self.steps:
                status = status and step.run((argset, obsset), runtime=runtime)[0]  # TODO iterate correctly
                if not status:
                    raise RuntimeError(f"Failed to run step=[{step.name}]")
            out.append(status)
        return out

    def fetch(self, *args: RunSet, runtime: Runtime, observables: ObsQuery | None = None) -> Sequence[ObsMap]:
        # TODO verify this
        out = list[ObsMap]()
        for argset, obsset in args:
            print(f"experiment=[{self.name}].fetch(), argset={tostr(argset)}, obsset={tostr(obsset)}")
            cond = lambda c: True if observables is None else c.typing in observables
            O = [c for c in obsset if cond(c)]
            for step in self.steps:
                # TODO verify this
                result = step.fetch((argset, O), runtime=runtime)[0]
                # merge parameters and observables
                out.append(result)
        return out
