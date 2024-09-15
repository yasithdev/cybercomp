from typing import TypeVar, Generic


class Type:
    """
    Base class for a semantic type

    """


class Category:
    """
    Base class for a category of semantic types

    """


T = TypeVar("T")


class Parameter(Generic[T]):
    """
    Base class for a parameter

    """

    value: T

    def __init__(self, value: T) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return str(self.value)


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
