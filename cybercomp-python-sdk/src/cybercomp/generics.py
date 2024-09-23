from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T", contravariant=True)


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
    typing: TypeVar

    def with_value(self, value: T) -> Parameter[T]:
        self.value = value
        return self

    def with_typing(self, tv: TypeVar) -> Parameter[T]:
        self.typing = tv
        return self


class RequiredParameter(Parameter[T]):
    """
    Base class for a required parameter

    """

    required = True

    def with_value(self, value: T) -> RequiredParameter[T]:
        self.value = value
        return self


class OptionalParameter(Parameter[T]):
    """
    Base class for an optional parameter

    """

    required = False

    def with_value(self, value: T) -> OptionalParameter[T]:
        self.value = value
        return self


class Hyperparameter(Generic[T]):
    """
    Base class for a hyperparameter

    """

    value: T
    typing: TypeVar

    def with_value(self, value: T) -> Hyperparameter[T]:
        self.value = value
        return self

    def with_typing(self, tv: TypeVar) -> Hyperparameter[T]:
        self.typing = tv
        return self


class Observation(Generic[T]):
    """
    Base class for an observation

    """

    value: T
    typing: TypeVar

    def with_value(self, value: T) -> Observation[T]:
        self.value = value
        return self

    def with_typing(self, tv: TypeVar) -> Observation[T]:
        self.typing = tv
        return self
