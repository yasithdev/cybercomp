from __future__ import annotations

from itertools import chain, product
from typing import Sequence, Set

from cybercomp.base import ArgSet, RunSet, ObsSet, ObsMap

from .base import ArgSet, Engine, Hyperparameter, Model, Observable, Parameter, Runnable, Runtime


def args(
    parameter_space: Sequence[Set[Parameter]],
    hyperparameter_space: Sequence[Set[Hyperparameter]],
) -> Sequence[ArgSet]:
    """
    Generate a set of arguments from a given parameter/hyperparameter space

    @param parameter_space: a sequence of parameter sets.
                            all items have the same set of parameters,
                            but with different values.
    @hyperparameter_space:  a sequence of hyperparameter sets.
                            all items have the same set of hyperparameters,
                            but with different values.

    """
    # cross product of parameter and hyperparameter spaces
    argsets = list[ArgSet]()
    for parameters, hyperparameters in product(parameter_space, hyperparameter_space):
        argset = [*parameters, *hyperparameters]
        argsets.append(argset)
    return argsets


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
        return super().prepare(*args)
    
    def run(self, *args: RunSet, runtime: Runtime) -> Sequence[bool]:
        # TODO correctly implement this
        return [True] * len(args)
    
    def fetch(self, *args: RunSet, runtime: Runtime, observables: Sequence[Observable] | None = None) -> Sequence[ObsMap]:
        return super().fetch(*args, runtime=runtime, observables=observables)


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
        # TODO correctly implement this
        from .util_composition import merge_parameters_and_observables

        # do an independent run for each argset
        runsets = []
        for argset in args:
            # get the globally provided parameters, hyperparameters, and observables
            P = [c for c in argset if isinstance(c, Parameter)]
            H = [c for c in argset if isinstance(c, Hyperparameter)]
            O = []
            for step in self.steps:
                # chain these parameters with step-wise P,H,O
                spec = list(chain(P, H))
                O = step.run(spec, runtime=runtime)
                print(f"run [{step.name}] with P={P} H={H}")
                # merge parameters and observables
                P = merge_parameters_and_observables(P, O)
            runsets.append(O)
        return runsets

    def run(self, *args: RunSet, runtime: Runtime) -> Sequence[bool]:
        # TODO correctly implement this
        return [True] * len(args)

    def fetch(self, *args: RunSet, runtime: Runtime, observables: ObsSet | None = None) -> Sequence[ObsMap]:
        # TODO correctly implement this
        out = []
        for argset in args:
            # get the globally provided parameters, hyperparameters, and observables
            P = [c for c in argset if isinstance(c, Parameter)]
            H = [c for c in argset if isinstance(c, Hyperparameter)]
            O = []
        for step in self.steps:
                # chain these parameters with step-wise P,H,O
                spec = list(chain(P, H))
                O = step.run(spec, runtime=runtime)
                print(f"run [{step.name}] with P={P} H={H}")
                # merge parameters and observables
                P = merge_parameters_and_observables(P, O)
            out.append(O)
        return out


# from .exceptions import TypeMismatchError

# conditions: ArgSet = [],

# def describe(self, P, H, O):
#     print(f"[Experiment] {self.name} is created")
#     p_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in P])
#     print(f"[P] {p_str}")
#     h_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in H])
#     print(f"[H] {h_str}")
#     o_str = "\n".join([f"\t{x.typing.__name__}={x.value}" for x in O])
#     print(f"[O] {o_str}")

# P = create_parameters(context, parameters)
# H = create_hyperparameters(context, hyperparameters)
# O = create_observables(context, observables)

# def create_parameters(context: Step | Experiment, parameters: Parameters) -> Sequence[Parameter]:
#     if isinstance(context, Step):
#         M = context.model
#         names = M.__dir__()
#         obj_name = M.__class__.__name__
#     else:
#         names = set()
#         objs = []
#         for step in context.steps:
#             assert isinstance(step, Step)
#             names.update([p.typing.__name__ for p in step.P])
#             objs.append(step.name)
#         obj_name = "|".join(objs)

#     P = list[Parameter]()
#     for p in parameters:
#         tv, value = p.typing, p.value
#         if tv.__name__ not in names:
#             raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{obj_name}'")
#             # ensure this parameter exists
#         P.append(p)
#     return P


# def create_hyperparameters(context: Step | Experiment, hyperparameters: Hyperparameters) -> Sequence[Hyperparameter]:
#     if isinstance(context, Step):
#         E = context.engine
#         names = E.__dir__()
#         obj_name = E.__class__.__name__
#     else:
#         names = set()
#         objs = []
#         for step in context.steps:
#             assert isinstance(step, Step)
#             names.update([h.typing.__name__ for h in step.H])
#             objs.append(step.name)
#         obj_name = "|".join(objs)
#     H = list[Hyperparameter]()
#     for h in hyperparameters:
#         tv, value = h.typing, h.value
#         if tv.__name__ not in names:
#             raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{obj_name}'")
#             # ensure this hyperparameter exists
#         H.append(h)
#     return H


# def create_observables(context: Step | Experiment, observables: Observables) -> Sequence[Observable]:
#     if isinstance(context, Step):
#         M = context.model
#         names = M.__dir__()
#         obj_name = M.__class__.__name__
#     else:
#         names = set()
#         objs = []
#         for step in context.steps:
#             assert isinstance(step, Step)
#             names.update([o.typing.__name__ for o in step.O])
#             objs.append(step.name)
#         obj_name = "|".join(objs)

#     O = list[Observable]()
#     for o in observables:
#         tv, value = o.typing, o.value
#         if tv.__name__ not in names:
#             raise TypeMismatchError(f"Parameter '{tv.__name__}' does not exist in '{obj_name}'")
#             # ensure this observation exists
#         O.append(o)
#     return O
