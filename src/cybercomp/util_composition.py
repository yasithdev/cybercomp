from .base import Args, DefaultParameter, Hyperparameter, Obs, Params, Hparams, Observable, Parameter


def as_args(obs: Obs) -> Args:
    args = set[Parameter | Hyperparameter]()
    for ob in obs:
        t = ob.typing
        v = ob.value if ob.initialized else "some/autogenerated/value"
        args.add(Parameter(t)(v))
    return args


def as_obs(args: Args) -> Obs:
    obs = set[Observable]()
    for arg in args:
        t = arg.typing
        v = arg.value if arg.initialized else "some/autogenerated/value"
        obs.add(Observable(t)(v))
    return obs


def union_a(a: Args, o: Obs) -> Args:
    """
    Combine arguments with observables, return as arguments

    @param a: a set of arguments
    @param o: a set of observables
    @return: the next argument set

    """
    return a.union(as_args(o))


def union_o(a: Args, o: Obs) -> Obs:
    """
    Combine arguments with observables, return as observables

    @param a: a set of arguments
    @param o: a set of observables
    @return: the next argument set

    """
    return as_obs(a).union(o)


def int_p(a: Params, o: Obs) -> Params:
    """
    Create the next param set by intersecting the current param set with the observable set

    @param a: a set of params
    @param o: a set of observables
    @return: the intersection as params

    """
    return a.intersection(as_args(o))


def int_h(a: Hparams, o: Obs) -> Hparams:
    """
    Create the next hparam set by intersecting the current hparam set with the observable set

    @param a: a set of hparams
    @param o: a set of observables
    @return: the intersection as hparams

    """
    return a.intersection(as_args(o))


def int_o(a: Args, o: Obs) -> Obs:
    """
    Create the next observable set by intersecting the current observable set with the argument set

    @param a: the current argument set
    @param o: the observable set
    @return: the next observable set
    """
    return as_obs(a).intersection(o)


def split(uP: Params, uH: Hparams, uO: Obs) -> tuple[tuple[Params, Hparams], tuple[Params, Hparams], Obs]:

    # NOTE ignoring default parameters from uP
    uP = {p for p in uP if not isinstance(p, DefaultParameter)}

    iP = int_p(uP, uO)
    iH = int_h(uH, uO)
    iO = int_o(uP.union(uH), uO)

    return (
        (uP.difference(iP), uH.difference(iH)),
        (iP, iH),
        uO.difference(iO),
    )