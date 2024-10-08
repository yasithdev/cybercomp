from typing import Sequence

from .base import Any, Observable, Parameter, TypeVar


def merge_parameters_and_observables(P1: Sequence[Parameter], O1: Sequence[Observable]) -> Sequence[Parameter]:
    out = set[tuple[TypeVar, Any]]()
    for p1 in P1:
        out.add((p1.typing, p1.value))
    for o1 in O1:
        out.add((o1.typing, o1.value))
    return [Parameter[t](t, v) for t, v in out]
