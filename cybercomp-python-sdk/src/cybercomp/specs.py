from pydantic import BaseModel

primitives = dict(
    file="str",
    binary="str",
    archive="str",
)


class TypeSpec(BaseModel):
    """
    Specification for a semantic type

    """

    description: str
    form: str


SourceSpec = str
"""
Specification for a source code

"""


CategorySpec = dict[str, TypeSpec]
"""
Specification for a category of semantic types

"""


class RecipeSpec(BaseModel):
    """
    Specification for a recipe to run a (model, experiment) pair

    """

    command: list[str]


class EngineSpec(BaseModel):
    """
    Specification for an execution engine (e.g., ODE solver) to simulate models

    """

    description: str
    source_id: str
    engine_parameters: dict[str, str]
    supported_models: dict[str, RecipeSpec]


class ModelSpec(BaseModel):
    """
    A computational model, described by parameters and observations

    """

    description: str
    required_parameters: dict[str, str]
    optional_parameters: dict[str, str]
    observations: dict[str, str]

    def validate(self, types: dict[str, TypeSpec]):
        # check if required parameters match types
        for param, form in self.required_parameters.items():
            # validate if type is defined
            if param not in types:
                raise TypeError(f"type={param} does not exist")
            # validate if type spec matches
            typ = types[param]
            if form != typ.form:
                raise TypeError(f"format={form} does not match the defined type={typ}")
        print("[Model] Validated")
