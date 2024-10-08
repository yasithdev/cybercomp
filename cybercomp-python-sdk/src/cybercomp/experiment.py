from __future__ import annotations

from itertools import chain, product
from typing import Sequence

from .base import Engine, Hyperparameter, Model, Observable, Observation, Parameter, Runnable, Runtime
from .util_composition import merge_hyperparameters, merge_observables, merge_parameters_and_observables

ArgSet = Sequence[Parameter | Hyperparameter | Observable]


def args(
    parameter_space: Sequence[Sequence[Parameter]],
    hyperparameter_space: Sequence[Sequence[Hyperparameter]],
    observables: Sequence[Observable],
) -> Sequence[ArgSet]:
    """
    Replicate a computational step on discrete samples from
    a parameter/hyperparameter space.

    """
    # cross product of parameter and hyperparameter spaces
    argsets = list[ArgSet]()
    for parameters, hyperparameters in product(parameter_space, hyperparameter_space):
        argset = [*parameters, *hyperparameters, *observables]
        argsets.append(argset)
    return argsets


def step(model: Model, engine: Engine, name: str) -> Step:
    return Step(name, model, engine)


def experiment(*steps: Step, name: str) -> Experiment:
    return Experiment(name, *steps)


def replicate(step: Step, *argset: ArgSet, name: str) -> Experiment:
    return Experiment(name, step).bind(*argset)


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

    def __rshift__(self, other: Runnable) -> Experiment:
        if not isinstance(other, Runnable):
            raise TypeError(f"Unsupported operand type(s) for >>: 'Step' and '{type(other).__name__}'")
        return Experiment(f"{self.name}>>{other.name}", self, other)

    def run(self, *args: ArgSet, runtime: Runtime) -> bool:
        return super().run(*args, runtime=runtime)

    def fetch(
        self, *args: ArgSet, runtime: Runtime, observables: Sequence[Observable]
    ) -> Sequence[Sequence[Observation]]:
        return super().fetch(*args, runtime=runtime, observables=observables)


class Experiment(Runnable):
    """
    A sequence of computational steps where
    the next step depends on past outputs.

    """

    specs: list[ArgSet] = []

    def __init__(self, name: str, *steps: Runnable) -> None:
        assert len(steps) > 0
        super().__init__(name)
        self.steps = steps

    def __rshift__(self, other: Step) -> Experiment:
        if not isinstance(other, Step):
            raise TypeError(f"Unsupported operand type(s) for >>: 'Experiment' and '{type(other).__name__}'")
        return Experiment(f"{self.name}>>{other.name}", self, other)

    def __str__(self) -> str:
        return f"name=[{self.name}], steps=[" + ">>".join([s.name for s in self.steps]) + "]"

    def bind(self, *args: ArgSet) -> Experiment:
        """
        Bind runtime args to the collection

        """
        P, H, O = [], [], []
        for i, A in enumerate(args):
            P = merge_parameters_and_observables(P, O, [c for c in A if isinstance(c, Parameter)])
            H = merge_hyperparameters(H, [c for c in A if isinstance(c, Hyperparameter)])
            O = merge_observables(O, [c for c in A if isinstance(c, Observable)])
            spec = list(chain(P, H, O))
            self.specs.append(spec)
            print(f"[{self.name}][{i}] {P} {H} {O}")
        return self

    def run(self, *args: ArgSet, runtime: Runtime) -> bool:
        return super().run(*args, runtime=runtime)

    def fetch(
        self, *args: ArgSet, runtime: Runtime, observables: Sequence[Observable]
    ) -> Sequence[Sequence[Observation]]:
        return super().fetch(*args, runtime=runtime, observables=observables)


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
