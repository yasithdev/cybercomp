from __future__ import annotations

from typing import Iterator

from cybercomp.base import ArgSet, Observable, ObsMap, ObsQuery, RunSet

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

    def prepare(self, args: ArgSet, level: int = 0) -> RunSet:
        prefix = "  " * level
        print(f"{prefix}->[Step] {self.name}.prepare()")
        print(tostr("args", args, level + 1))
        # first, ensure that args satisfy model and engine requirements
        # next, concatenate the args with model observables and return it
        (param_req, param_opt, observ) = self.model.describe(level + 1)
        hparam = self.engine.describe(level + 1)
        print(f"{prefix}  param_req={param_req}")
        print(f"{prefix}  param_opt={param_opt}")
        print(f"{prefix}  observ={observ}")
        print(f"{prefix}  hparam={hparam}")
        print()
        return RunSet(args={*param_req, *param_opt, *hparam}, obs=observ)

    def run(self, run: RunSet, runtime: Runtime, level: int = 0) -> RunState:
        prefix = "  " * level
        print(f"{prefix}->[Step] {self.name}.run()")
        print(tostr("args=", run.args, level + 1))
        print(tostr("obs=", run.obs, level + 1))
        print()
        return runtime.run(self.model, self.engine, run)

    def poll(self, run: RunSet, runtime: Runtime, level: int = 0) -> RunState:
        prefix = "  " * level
        print(f"{prefix}->[Step] {self.name}.run()")
        print(tostr("args=", run.args, level + 1))
        print(tostr("obs=", run.obs, level + 1))
        print()
        return runtime.poll(self.model, self.engine, run)

    def fetch(self, run: RunSet, runtime: Runtime, query: ObsQuery = None, level: int = 0) -> ObsMap:
        prefix = "  " * level
        print(f"{prefix}->[Step] {self.name}.fetch()")
        print(tostr("args=", run.args, level + 1))
        print(tostr("obs=", run.obs, level + 1))
        print()
        subquery = set()
        for key in self.__dir__():
            attr = getattr(self, key)
            if isinstance(attr, Observable):
                if query is None or attr.typing in {o.typing for o in query}:
                    subquery.add(attr)
        return runtime.fetch(self.model, self.engine, run, subquery)

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

    def prepare(self, args: ArgSet, level: int = 0) -> RunSet:
        """
        Prepare the experiment by identifying the
        inputs (argsets) and outputs (obsets) for the experiment.
        """
        from .util_composition import split, union_a

        # do an independent run for each argset
        prefix = "  " * level
        print(f"{prefix}->[Experiment] {self.name}.prepare()")
        # setup iteration variables
        uA = set()
        uO = set()
        for step in self.steps:
            # step_args is the subset of args that's used by step
            (a, o) = step.prepare(args, level=level + 1)
            uA.update(a)
            uO.update(o)
            args = union_a(args, o)

        # validate external args, reused, and external obs
        external_args, reused, external_obs = split(uA, uO)
        assert len(external_args.difference(args)) == 0, "Some external args are not provided"
        assert len(reused) > 0, "Experiment has mutually exclusive steps"
        assert uO.intersection(external_obs) == external_obs, "Experiment generates no external obs"

        # return (essential args, and generated obs
        return RunSet(args=external_args, obs=uO)

    def describe(self, level: int = 0) -> None:
        prefix = "  " * level
        print(f"{prefix}->[Experiment] {self.name}")
        print()
        for step in self.steps:
            step.describe(level + 1)

    def run(self, run: RunSet, runtime: Runtime, level: int = 0) -> Iterator[RunState]:
        prefix = "  " * level
        # performing an end-to-end experiment run on a single runset
        print(f"{prefix}->[Experiment] {self.name}.run()")
        for step in self.steps:
            # check if the step was successfully queued
            if isinstance(step, Experiment):
                # run each step in sequence
                for state in step.run(run, runtime=runtime, level=level + 1):
                    if state != "QUEUED":
                        raise RuntimeError(f"Failed step=[{step.name}]")
            elif isinstance(step, Step):
                state = step.run(run, runtime=runtime, level=level + 1)
                if state != "QUEUED":
                    raise RuntimeError(f"Failed to queue step=[{step.name}]")
            else:
                raise ValueError(f"Unsupported step type=[{type(step).__name__}]")
            # append the state to the status list
            yield "QUEUED"

    def poll(self, run: RunSet, runtime: Runtime, level: int = 0) -> Iterator[RunState]:
        prefix = "  " * level
        print(f"{prefix}->[Experiment] {self.name}.run()")
        for step in self.steps:
            # check if the step was successfully queued
            if isinstance(step, Experiment):
                for state in step.poll(run, runtime=runtime, level=level + 1):
                    if state != "COMPLETED":
                        raise RuntimeError(f"Failed to complete step=[{step.name}]")
            elif isinstance(step, Step):
                state = step.poll(run, runtime=runtime, level=level + 1)
                if state != "COMPLETED":
                    raise RuntimeError(f"Failed to complete step=[{step.name}]")
            else:
                raise ValueError(f"Unsupported step type=[{type(step).__name__}]")
            yield state

    def fetch(self, run: RunSet, runtime: Runtime, query: ObsQuery = None, level: int = 0) -> ObsMap:
        prefix = "  " * level
        print(f"{prefix}->[Experiment] {self.name}.fetch()")
        obs = dict()
        for step in self.steps:
            result = step.fetch(run, runtime=runtime, level=level + 1)
            obs.update(result)
        return obs
