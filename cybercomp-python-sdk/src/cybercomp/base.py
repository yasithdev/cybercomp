from .generics import Hyperparameter, Observation, Parameter


class Model:
    """
    Base class for a Computational Model
    """

    def parameters(self):
        out: dict[str, Parameter] = {}
        for attr in dir(self):
            v = getattr(self, attr)
            if isinstance(v, Parameter):
                out[attr] = v
        return out

    def observations(self):
        out: dict[str, Observation] = {}
        for attr in dir(self):
            v = getattr(self, attr)
            if isinstance(v, Observation):
                out[attr] = v
        return out

    def with_observations(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)
        return self


class Engine:
    """
    Base class for a Computational Engine
    """

    def hyperparameters(self):
        out: dict[str, Hyperparameter] = {}
        for attr in dir(self):
            v = getattr(self, attr)
            if isinstance(v, Hyperparameter):
                out[attr] = v
        return out


class Runtime:
    """
    Base class for an execution context
    """

    pass
