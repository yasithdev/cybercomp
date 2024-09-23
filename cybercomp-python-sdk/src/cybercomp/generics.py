from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Type(Generic[T]):
    """
    Base class for a semantic type

    """


class Category:
    """
    Base class for a category of semantic types

    """


class Parameter(Generic[T]):
    """
    Base class for a parameter

    """

    value: T
    required: bool

    def __init__(self, value: T) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return str(self.value)


class RequiredParameter(Parameter[T]):
    """
    Base class for a required parameter

    """

    required = True


class OptionalParameter(Parameter[T]):
    """
    Base class for an optional parameter

    """

    required = False


class Hyperparameter(Generic[T]):
    """
    Base class for a hyperparameter

    """

    value: T

    def __init__(self, value: T) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return str(self.value)


class Observation(Generic[T]):
    """
    Base class for an observation

    """

    value: T

    def __init__(self, value: T) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return str(self.value)
