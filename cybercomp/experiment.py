from .cybercomp import Model, Engine


class Collection:

    models: tuple[Model, ...]

    def __init__(self, *models: Model) -> None:
        self.models = models


class Experiment:
    """class to define a concrete experiment, with concrete input/output values provided"""

    name: str
    model: Model | Collection
    engine: Engine

    def validate(self):
        is_model_valid = self.model.validate(self.config.inputs, self.config.observations)
        is_engine_valid = self.engine.validate(self.config.hyperparams)
        return is_model_valid and is_engine_valid

    def execute(self):
        is_valid = self.validate()
        if not is_valid:
            raise ValueError("run_config does not match model and engine")
        else:
            self.run()

    def run(self):
        """
        Does nothing for now
        Function should invoke the run

        """
        print("run executed")
