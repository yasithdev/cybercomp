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

    def prepare(self, *args: ArgSet, level: int = 0) -> Sequence[RunSet]:
        out = []
        for argset in args:
            print(f"[Step] {self.name}.prepare()", tostr("argset", argset))
            # first, ensure that argset satisfy model and engine requirements
            # next, concatenate the argset with model observables and return it
            (param_req, param_opt, observ) = self.model.describe(level + 1)
            hparam = self.engine.describe(level + 1)
            print(f"param_req={param_req}, param_opt={param_opt}, observ={observ}, hparam={hparam}")
            out.append(([*param_req, *param_opt], observ))
        return out

    def run(self, *args: RunSet, runtime: Runtime) -> Sequence[bool]:
        out = []
        for argset, obsset in args:
            print(f"[Step] {self.name}.run()", tostr("argset", argset), tostr("obsset", obsset))
            status = runtime.run(self.model, self.engine, argset, obsset)
            out.append(status)
        return out

    def fetch(self, *args: RunSet, runtime: Runtime, **_) -> Sequence[ObsMap]:
        out = []
        for argset, obsset in args:
            print(f"[Step] {self.name}.fetch()", tostr("argset", argset), tostr("obsset", obsset))
            obsmap = runtime.fetch(self.model, self.engine, argset, obsset)
            out.append(obsmap)
        return out

    def describe(self, level: int = 0) -> None:
        prefix = "  " * level
        print(f"{prefix[:-1]}⮑[Step] {self.name}")
        print(f"{prefix}* model={self.model.__class__.__name__}")
        print(f"{prefix}* engine={self.engine.__class__.__name__}")
        print()
        self.model.describe(level + 1)
        self.engine.describe(level + 1)


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

    def prepare(self, *args: ArgSet, level: int = 0) -> Sequence[RunSet]:
        # TODO improve loop structure
        from .util_composition import create_next_argset, create_next_obsset

        # do an independent run for each argset
        runsets = list[RunSet]()
        for prev_argset in args:
            print(f"[Experiment] {self.name}.prepare()")
            # no observations in the beginning
            next_argset = prev_argset
            prev_obsset = []
            for step in self.steps:
                prev_argset = next_argset
                # step_argset is the subset of args that's used by step
                (step_argset, step_obsset) = step.prepare(prev_argset, level=level + 1)[0]
                next_argset = create_next_argset(prev_argset, step_obsset)
                next_obsset = create_next_obsset(prev_obsset, step_obsset)
            # append final runset
            runsets.append((prev_argset, next_obsset))

        return runsets

    def describe(self, level: int = 0) -> None:
        prefix = "  " * level
        print(f"{prefix[:-1]}⮑[Experiment] {self.name}")
        print()
        for step in self.steps:
            step.describe(level + 1)

    def run(self, *args: RunSet, runtime: Runtime) -> Sequence[bool]:
        # TODO improve loop structure
        out = []
        for argset, obsset in args:
            print(f"[Experiment] {self.name}.run()")
            status = True
            for step in self.steps:
                status = status and step.run((argset, obsset), runtime=runtime)[0]
                if not status:
                    raise RuntimeError(f"Failed to run step=[{step.name}]")
            out.append(status)
        return out

    def fetch(self, *args: RunSet, runtime: Runtime, observables: ObsQuery | None = None) -> Sequence[ObsMap]:
        out = list[ObsMap]()
        for argset, obsset in args:
            print(f"[Experiment] {self.name}.fetch()")
            cond = lambda c: True if observables is None else c in observables
            O = [c for c in obsset if cond(c)]
            argset_obs = dict()
            for step in self.steps:
                result = step.fetch((argset, O), runtime=runtime)[0]
                argset_obs.update(result)
            out.append(argset_obs)
        return out
