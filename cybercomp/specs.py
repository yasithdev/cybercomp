from pydantic import BaseModel


class TypeSpec(BaseModel):
    """
    A single semantic type

    """

    formats: list[str]


class TypeGroupSpec(BaseModel):
    """
    A group of semantic types

    """

    category: str
    types: dict[str, TypeSpec]


class RecipeSpec(BaseModel):
    command: list[str]


class EngineSpec(BaseModel):
    """
    A computational engine (i.e., the execution program)

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

    def validate(self, type_groups: dict[str, TypeGroupSpec]):
        # flatten the types into dict
        flattened_types: dict[str, list[str]] = dict()
        for group in type_groups.values():
            for typ, spec in group.types.items():
                flattened_types[f"{group.category}_{typ}"] = spec.formats
        # check if required parameters match types
        for param, form in self.required_parameters.items():
            # validate if type is defined
            if param not in flattened_types:
                raise TypeError(f"type={param} does not exist")
            # validate if type spec matches
            if form not in flattened_types[param]:
                raise TypeError(f"format={form} does not match the defined type={param}")
        print("[Model] Validated")
