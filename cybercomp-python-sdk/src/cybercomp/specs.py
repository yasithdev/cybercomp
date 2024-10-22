from typing import Any

from pydantic import BaseModel, Field

primitives = dict(
    path="str",
    string="str",
    env="str",
    numeric="float",
)


def get_canonical_type(v: str) -> str:
    typ = v
    if v in primitives:
        typ = primitives[v]
    return typ


class TypeSpec(BaseModel):
    """
    Specification for an argument with a type and optional default

    """

    type: str
    default: Any = Field(None)


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

    engine: list[str] = Field([])
    "the list of engines this recipe depends on"
    command: str
    "the templated command to run the recipe"


class EngineSpec(BaseModel):
    """
    Specification for an execution engine (e.g., ODE solver) to simulate models

    """

    description: str
    parameters: dict[str, TypeSpec] = Field({})


class ModelSpec(BaseModel):
    """
    A computational model, described by parameters and observables

    """

    description: str
    parameters: dict[str, TypeSpec] = Field([])
    observables: dict[str, TypeSpec] = Field([])
    run: RecipeSpec

    def validate(self, specs: dict[str, TypeSpec]):
        # check if required parameters match types
        for key, spec in self.parameters.items():
            # validate if type is defined
            if key not in specs:
                specs[key] = spec
                # raise TypeError(f"type={param} does not exist")
            # validate if type spec matches
            spec_type = get_canonical_type(spec.type)
            slot_type = get_canonical_type(specs[key].type)
            if spec_type != slot_type:
                raise TypeError(f"parameter {key}: {spec_type} does not match type={slot_type}")
        print("[Model] Validated")
