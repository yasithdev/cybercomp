from __future__ import annotations
from typing import Any, TypeVar, Generic

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

    def __init__(self, value: T) -> None:
        super().__init__()
        self.value = value


class Hyperparameter(Generic[T]):
    """
    Base class for a hyperparameter

    """

    value: T

    def __init__(self, value: T) -> None:
        super().__init__()
        self.value = value


class Observation(Generic[T]):
    """
    Base class for an observation

    """

    value: T

    def __init__(self, value: T) -> None:
        super().__init__()
        self.value = value
