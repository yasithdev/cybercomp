from __future__ import annotations

from typing import Sequence

from cybercomp.base import ArgSet, ObsMap, ObsQuery, RunSet

from .base import ArgSet, Engine, Model, Runnable, RunState, Runtime, tostr


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
        out = list[RunSet]()
        prefix = "  " * level
        for argset in args:
            print(f"{prefix}->[Step] {self.name}.prepare()")
            print(tostr("argset", argset, level + 1))
            # first, ensure that argset satisfy model and engine requirements
            # next, concatenate the argset with model observables and return it
            (param_req, param_opt, observ) = self.model.describe(level + 1)
            hparam = self.engine.describe(level + 1)
            print(f"{prefix}  param_req={param_req}")
            print(f"{prefix}  param_opt={param_opt}")
            print(f"{prefix}  observ={observ}")
            print(f"{prefix}  hparam={hparam}")
            print()
            out.append(([*param_req, *param_opt, *hparam], observ))
        return out

    def run(self, *args: RunSet, runtime: Runtime, level: int = 0) -> Sequence[RunState]:
        out = list[RunState]()
        prefix = "  " * level
        for argset, obsset in args:
            print(f"{prefix}->[Step] {self.name}.run()")
            print(tostr("argset", argset, level + 1))
            print(tostr("obsset", obsset, level + 1))
            print()
            status = runtime.run(self.model, self.engine, argset, obsset)
            out.append(status)
        return out
    
    def poll(self, *args: RunSet, runtime: Runtime, level: int = 0) -> Sequence[RunState]:
        out = list[RunState]()
        prefix = "  " * level
        for argset, obsset in args:
            print(f"{prefix}->[Step] {self.name}.run()")
            print(tostr("argset", argset, level + 1))
            print(tostr("obsset", obsset, level + 1))
            print()
            status = runtime.poll(self.model, self.engine, argset, obsset)
            out.append(status)
        return out

    def fetch(self, *args: RunSet, runtime: Runtime, level: int = 0, **_) -> Sequence[ObsMap]:
        out = list[ObsMap]()
        prefix = "  " * level
        for argset, obsset in args:
            print(f"{prefix}->[Step] {self.name}.fetch()")
            print(tostr("argset", argset, level + 1))
            print(tostr("obsset", obsset, level + 1))
            print()
            obsmap = runtime.fetch(self.model, self.engine, argset, obsset)
            out.append(obsmap)
        return out

    def describe(self, level: int = 0) -> None:
        prefix = "  " * level
        print(f"{prefix}->[Step] {self.name}")
        print(f"{prefix}  model={self.model.__class__.__name__}")
        print(f"{prefix}  engine={self.engine.__class__.__name__}")
        print()
        self.model.describe(level + 1)
        self.engine.describe(level + 1)


class Experiment(Runnable):
    """
    A sequence of computational steps where
    each step depends on the previous step(s).

    """

    def __init__(self, name: str, *steps: Step | Experiment) -> None:
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
        prefix = "  " * level
        runsets = list[RunSet]()
        for prev_argset in args:
            print(f"{prefix}->[Experiment] {self.name}.prepare()")
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
        print(f"{prefix}->[Experiment] {self.name}")
        print()
        for step in self.steps:
            step.describe(level + 1)

    def run(self, *args: RunSet, runtime: Runtime, level: int = 0) -> Sequence[Sequence[RunState]]:
        # TODO improve loop structure
        out = list[list[RunState]]()
        prefix = "  " * level
        for argset, obsset in args:
            print(f"{prefix}->[Experiment] {self.name}.run()")
            status = list[RunState]()
            for step in self.steps:
                # check if the step was successfully queued
                if isinstance(step, Experiment):
                    state = step.run((argset, obsset), runtime=runtime, level=level + 1)[0]
                    for s in state:
                        if s != "QUEUED":
                            raise RuntimeError(f"Failed to queue step=[{step.name}]")
                elif isinstance(step, Step):
                    state = step.run((argset, obsset), runtime=runtime, level=level + 1)[0]
                    if state != "QUEUED":
                        raise RuntimeError(f"Failed to queue step=[{step.name}]")
                else:
                    raise ValueError(f"Unsupported step type=[{type(step).__name__}]")
                # append the state to the status list
                status.append("QUEUED")
            out.append(status)
        return out

    def poll(self, *args: RunSet, runtime: Runtime, level: int = 0) -> Sequence[Sequence[RunState]]:
        # TODO improve loop structure
        out = []
        prefix = "  " * level
        for argset, obsset in args:
            print(f"{prefix}->[Experiment] {self.name}.run()")
            status = list[RunState]()
            for step in self.steps:
                # check if the step was successfully queued
                if isinstance(step, Experiment):
                    state = step.poll((argset, obsset), runtime=runtime, level=level + 1)[0]
                    for s in state:
                        if s != "COMPLETED":
                            raise RuntimeError(f"Failed to complete step=[{step.name}]")
                elif isinstance(step, Step):
                    state = step.poll((argset, obsset), runtime=runtime, level=level + 1)[0]
                    if state != "COMPLETED":
                        raise RuntimeError(f"Failed to complete step=[{step.name}]")
                else:
                    raise ValueError(f"Unsupported step type=[{type(step).__name__}]")
            out.append(status)
        return out

    def fetch(self, *args: RunSet, runtime: Runtime, observables: ObsQuery = None, level: int = 0) -> Sequence[ObsMap]:
        out = list[ObsMap]()
        prefix = "  " * level
        for argset, obsset in args:
            print(f"{prefix}->[Experiment] {self.name}.fetch()")
            cond = lambda c: True if observables is None else c in observables
            O = [c for c in obsset if cond(c)]
            argset_obs = dict()
            for step in self.steps:
                result = step.fetch((argset, O), runtime=runtime, level=level + 1)[0]
                argset_obs.update(result)
            out.append(argset_obs)
        return out
