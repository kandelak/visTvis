"""
Decorator utilities for visTvis.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from .state import state
from .storage import store_value


class LayerCounter:
    """
    Small int-like helper that keeps the visTvis state in sync while it is
    mutated. Use in place of an integer index and prefer in-place operations
    such as ``counter += 1`` so that the state stays consistent.
    """

    def __init__(self, identifier: str, start: int) -> None:
        self.identifier = identifier
        self.value = int(start) if start is not None else 0
        state.update(identifier, self.value)

    def _sync(self) -> None:
        state.update(self.identifier, self.value)

    def __iadd__(self, other: Any):
        self.value += int(other)
        self._sync()
        return self

    def __isub__(self, other: Any):
        self.value -= int(other)
        self._sync()
        return self

    def __int__(self) -> int:  # pragma: no cover - trivial
        return self.value

    def __index__(self) -> int:  # pragma: no cover - trivial
        return self.value

    def __add__(self, other: Any) -> int:
        return self.value + int(other)

    def __radd__(self, other: Any) -> int:
        return int(other) + self.value

    def __sub__(self, other: Any) -> int:
        return self.value - int(other)

    def __rsub__(self, other: Any) -> int:
        return int(other) - self.value

    def __eq__(self, other: Any) -> bool:  # pragma: no cover - trivial
        try:
            return self.value == int(other)
        except Exception:
            return False

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"LayerCounter(identifier={self.identifier!r}, value={self.value})"


def visTvis_store(
        base_folder_path: str,
        identifier: str,
        file_name: str,
        folder_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Store a value returned by a function in a structured folder layout.

    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            layer = state.get_layer(identifier)
            store_value(result, base_folder_path, identifier, layer, file_name, folder_name)
            return result

        return wrapper

    return decorator


def visTvis_layer_counter(
        identifier: str,
        var_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Attach a LayerCounter to the decorated function so nested @visTvis_store
    calls know which layer they belong to.

    If the decorated function defines a parameter matching ``var_name`` the
    value is replaced with a LayerCounter instance that keeps the visTvis
    state in sync when it is mutated (e.g. ``counter += 1``). The final counter
    value is unwrapped back to a plain int in the return value.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)
        accepts_counter = var_name in signature.parameters

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_args = args
            call_kwargs = kwargs

            starting_value = state.get_layer(identifier)
            counter: Optional[LayerCounter] = None

            if accepts_counter:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                starting_value = bound.arguments.get(var_name, starting_value)
                counter = LayerCounter(identifier, int(starting_value))
                bound.arguments[var_name] = counter
                call_args = bound.args
                call_kwargs = bound.kwargs
            else:
                counter = LayerCounter(identifier, int(starting_value))

            token = state.push(identifier, int(counter))
            try:
                result = func(*call_args, **call_kwargs)
            finally:
                state.pop(token)

            state.update(identifier, int(counter))
            return _unwrap_layer_counters(result)

        return wrapper

    return decorator


def _unwrap_layer_counters(value: Any) -> Any:
    if isinstance(value, LayerCounter):
        return int(value)
    if isinstance(value, tuple):
        return tuple(_unwrap_layer_counters(v) for v in value)
    if isinstance(value, list):
        return [_unwrap_layer_counters(v) for v in value]
    if isinstance(value, dict):
        return {k: _unwrap_layer_counters(v) for k, v in value.items()}
    return value


__all__ = ["visTvis_store", "visTvis_layer_counter", "LayerCounter"]
