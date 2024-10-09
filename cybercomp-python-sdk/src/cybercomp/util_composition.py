from .base import ArgSet, Hyperparameter, ObsSet, Parameter


def create_next_argset(argset: ArgSet, obsset: ObsSet) -> ArgSet:
    """
    Create the next argument set by combining the current argument set with the observable set

    @param argset: the current argument set
    @param obsset: the observable set
    @return: the next argument set

    """
    P = [c for c in argset if isinstance(c, Parameter)]
    H = [c for c in argset if isinstance(c, Hyperparameter)]
    next_argset = P + [Parameter(c, None) for c in obsset] + H # TODO substitute none-valued placeholders with generated value
    return next_argset


def create_next_obsset(obsset: ObsSet, new_obsset: ObsSet) -> ObsSet:
    """
    Create the next observable set by combining the current observable set with the new observable set

    @param obsset: the current observable set
    @param new_obsset: the new observable set
    @return: the next observable set

    """
    return [*obsset, *new_obsset]
