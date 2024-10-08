from typing import Sequence

from .base import Any, Hyperparameter, Observable, Parameter, TypeVar


def merge_parameters_and_observables(
    P1: Sequence[Parameter], O1: Sequence[Observable], P2: Sequence[Parameter]
) -> Sequence[Parameter]:
    out = set[tuple[TypeVar, Any]]()
    for p1 in P1:
        out.add((p1.typing, p1.value))
    for o1 in O1:
        out.add((o1.typing, o1.value))
    for p2 in P2:
        out.add((p2.typing, p2.value))
    return [Parameter[t](t, v) for t, v in out]


def merge_hyperparameters(H1: Sequence[Hyperparameter], H2: Sequence[Hyperparameter]) -> Sequence[Hyperparameter]:
    out = set[tuple[TypeVar, Any]]()
    for h1 in H1:
        out.add((h1.typing, h1.value))
    for h2 in H2:
        out.add((h2.typing, h2.value))
    return [Hyperparameter[t](t, v) for t, v in out]


def merge_observables(O1: Sequence[Observable], O2: Sequence[Observable]) -> Sequence[Observable]:
    out = set[tuple[TypeVar, Any]]()
    for o1 in O1:
        out.add((o1.typing, o1.value))
    for o2 in O2:
        out.add((o2.typing, o2.value))
    return [Observable[t](t, v) for t, v in out]
