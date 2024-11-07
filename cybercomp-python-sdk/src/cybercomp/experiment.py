from __future__ import annotations

import time
from typing import Iterator

from cybercomp.base import Args, Observable, ObsMap, ObsQuery, RunConfig

from .base import Args, Engine, Hyperparameter, Model, Observable, Parameter, Runnable, RunState, Runtime


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

    def setup(self, args: Args, level: int = 0) -> Step:
        prefix = "  " * level
        print(f"{prefix}* [Step] {self.name}.setup()")
        # assign args
        P = {a for a in args if isinstance(a, Parameter)}
        H = {a for a in args if isinstance(a, Hyperparameter)}
        self.model.update(P)
        self.engine.update(H)
        (params, obs) = self.model.describe(level + 1)
        hparams = self.engine.describe(level + 1)
        self.config = RunConfig(params, hparams, obs)
        return self

    def run(self, runtime: Runtime, level: int = 0) -> RunState:
        assert self.config is not None, "Step.setup() must be called first"
        prefix = "  " * level
        print(f"{prefix}* [Step] {self.name}.run()")
        return runtime.run(self, level)

    def poll(self, runtime: Runtime, level: int = 0) -> RunState:
        assert self.config is not None, "Step.run() must be called first"
        prefix = "  " * level
        print(f"{prefix}* [Step] {self.name}.poll()")
        return runtime.poll(self, level)

    def fetch(self, runtime: Runtime, query: ObsQuery = None, level: int = 0) -> ObsMap:
        assert self.config is not None, "Step.run() must be called first"
        prefix = "  " * level
        print(f"{prefix}* [Step] {self.name}.fetch()")
        subquery = set()
        for key in self.__dir__():
            attr = getattr(self, key)
            if isinstance(attr, Observable):
                if query is None or attr.typing in {o.typing for o in query}:
                    subquery.add(attr)
        return runtime.fetch(self, subquery, level)

    def describe(self, level: int = 0) -> None:
        prefix = "  " * level
        print(f"{prefix}* [Step] {self.name}")
        print(f"{prefix}  model={self.model.__class__.__name__}")
        print(f"{prefix}  engine={self.engine.__class__.__name__}")
        self.model.describe(level + 1)
        self.engine.describe(level + 1)


class Experiment(Runnable):
    """
    An experiment is a sequence of executable items
    that could be either an invidual step (i.e. Step),
    or another experiment (i.e. Experiment).

    * X (expt)
        * X1 (step)
        * X2 (expt)
            * X2.1 (step)
            * X2.2 (expt)
                * X2.2.1 (step)
                * X2.2.2 (step)
            * X2.3 (step)
        * X3 (step)

    """

    def __init__(self, name: str, *steps: Step | Experiment) -> None:
        """
        Define an experiment with a name and a sequence of steps

        """
        assert len(steps) > 0
        super().__init__(name)
        self.steps = steps

    def __rshift__(self, other: Experiment) -> Experiment:
        """
        Combine two experiments into a single experiment
        in a non-mutating way.

        """
        if not isinstance(other, Experiment):
            raise TypeError(f"Unsupported operand type(s) for >>: 'Experiment' and '{type(other).__name__}'")
        return Experiment(f"{self.name}+{other.name}", self, other)

    def __str__(self) -> str:
        """
        Return a string representation of the experiment
        """
        return f"name=[{self.name}], steps=[" + ">>".join([s.name for s in self.steps]) + "]"

    def setup(self, args: Args, level: int = 0) -> Experiment:
        """
        Setup the experiment for the given args
        """
        from .util_composition import split, union_a

        # do an independent run for each Args
        prefix = "  " * level
        print(f"{prefix}* [Experiment] {self.name}.setup()")
        # setup iteration variables
        uP = set()
        uH = set()
        uO = set()
        for step in self.steps:
            # step_args is the subset of args that's used by step
            step.setup(args, level=level + 1)
            assert step.config is not None
            # ensure all observables are initialized, if not initialize them
            args = union_a(args, step.config.obs)
            uP.update(step.config.params)
            uH.update(step.config.hparams)
            uO.update(step.config.obs)

        # validate external args, reused, and external obs
        (ext_P, ext_H), (int_P, int_H), ext_O = split(uP, uH, uO)
        assert len(ext_P.difference(args)) == 0, "Some external params are not provided"
        assert len(ext_H.difference(args)) == 0, "Some external hparams are not provided"
        if len(self.steps) > 1:
            assert len(int_P) > 0, "Experiment has mutually exclusive steps"
        assert uO.intersection(ext_O) == ext_O, "Experiment generates no external obs"

        # update the configuration
        self.config = RunConfig(ext_P, ext_H, uO)

        # update the path in experiment steps
        for step in self.steps:
            assert step.config is not None
            step.config.dir_name = self.config.dir_name + step.config.dir_name
        return self

    def describe(self, level: int = 0) -> None:
        prefix = "  " * level
        print(f"{prefix}* [Experiment] {self.name}")
        for step in self.steps:
            step.describe(level + 1)

    def run(self, runtime: Runtime, level: int = 0) -> Iterator[RunState]:
        prefix = "  " * level
        # performing an end-to-end experiment run on a single RunConfig
        print(f"{prefix}* [Experiment] {self.name}.run()")
        for step in self.steps:
            # check if the step was successfully queued
            if isinstance(step, Experiment):
                # run each step in sequence
                for state in step.run(runtime=runtime, level=level + 1):
                    if state != "QUEUED":
                        raise RuntimeError(f"Failed step=[{step.name}]")
            elif isinstance(step, Step):
                state = step.run(runtime=runtime, level=level + 1)
                if state != "QUEUED":
                    raise RuntimeError(f"Failed to queue step=[{step.name}]")
            else:
                raise ValueError(f"Unsupported step type=[{type(step).__name__}]")
            # append the state to the status list
            yield "QUEUED"

    def run_sync(self, runtime: Runtime, dt: int = 1, level: int = 0) -> RunState:
        prefix = "  " * level
        print(f"{prefix}* [Experiment] {self.name}.run_sync()")

        # start the experiment
        for state in self.run(runtime=runtime, level=level + 1):
            if state != "QUEUED":
                raise RuntimeError(f"Failed to queue step=[{self.name}]")

        # wait for the experiment to complete
        run_state = "RUNNING"
        while run_state == "RUNNING":
            flags = []
            for state in self.poll(runtime=runtime):
                if state not in ["COMPLETED", "FAILED"]:
                    time.sleep(dt)
                    continue
                else:
                    flags.append(state == "COMPLETED")
            run_state = "COMPLETED" if all(flags) else "FAILED"
        return run_state

    def poll(self, runtime: Runtime, level: int = 0) -> Iterator[RunState]:
        prefix = "  " * level
        print(f"{prefix}* [Experiment] {self.name}.poll()")
        for step in self.steps:
            # check if the step was successfully queued
            if isinstance(step, Experiment):
                for state in step.poll(runtime=runtime, level=level + 1):
                    if state != "COMPLETED":
                        raise RuntimeError(f"Failed to complete step=[{step.name}]")
            elif isinstance(step, Step):
                state = step.poll(runtime=runtime, level=level + 1)
                if state != "COMPLETED":
                    raise RuntimeError(f"Failed to complete step=[{step.name}]")
            else:
                raise ValueError(f"Unsupported step type=[{type(step).__name__}]")
            yield state

    def fetch(self, runtime: Runtime, query: ObsQuery = None, level: int = 0) -> ObsMap:
        prefix = "  " * level
        print(f"{prefix}* [Experiment] {self.name}.fetch()")
        obs = dict()
        for step in self.steps:
            result = step.fetch(runtime=runtime, query=query, level=level + 1)
            obs.update(result)
        return obs
