from .base import Model, Engine


class Experiment:
    """class to define a concrete experiment, with concrete input/output values provided"""

    name: str
    model: Model
    engine: Engine
    parameters: dict[str, str]
    observables: dict[str, str]

    def execute(self):
        """
        Does nothing for now
        Function should invoke the run

        """
        print("run executed")


class Collection(Experiment):

    model: tuple[Model, ...]

    def __init__(self, *model: Model) -> None:
        self.model = model
