from typing import Sequence

from .base import Hyperparameter, Observation, Parameter


def observation_to_parameter(o: Observation) -> Parameter:
    return Parameter().with_value(o.value).with_typing(o.typing)


def merge_parameters_and_observations(
    P1: Sequence[Parameter], O1: Sequence[Observation], P2: Sequence[Parameter]
) -> Sequence[Parameter]:
    out: Sequence[Parameter] = []
    for p in P1:
        out.append(p)
    for o in O1:
        out.append(observation_to_parameter(o))
    for p in P2:
        # TODO properly implement
        if p in out:
            # when parameters are duplicated, enforce priority for newer one
            out.pop(out.index(p))
        out.append(p)
    return out


def merge_hyperparameters(H1: Sequence[Hyperparameter], H2: Sequence[Hyperparameter]) -> Sequence[Hyperparameter]:
    return [*H1, *H2]


def union_observations(O1: Sequence[Observation], O2: Sequence[Observation]) -> Sequence[Observation]:
    return [*O1, *O2]


def intersect_observations(O1: Sequence[Observation], O2: Sequence[Observation]) -> Sequence[Observation]:
    out: Sequence[Observation] = []
    for o in O2:
        # TODO properly implement
        if o in O1:
            out.append(o)
    return out
