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
    base_folder_path: str = "./vistvis_runs",
    identifier: str = "global",
    pos_ret: Optional[int] = None,
    var_name: str = "attn_weight",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Store a value returned by a function in a structured folder layout.

    The decorated function is executed normally, then the result is inspected to
    extract the value referenced by ``pos_ret`` (tuple/list index) or
    ``var_name`` (dictionary key or attribute). The value is saved to
    base_folder_path/identifier/layer_{layer}/{var_name}.pt using torch.save.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            value_to_store = _extract_value(result, var_name, pos_ret)
            layer = state.get_layer(identifier)
            store_value(value_to_store, base_folder_path, identifier, layer, var_name)
            return result

        return wrapper

    return decorator


def visTvis_layer_counter(
    identifier: str = "global",
    var_name: str = "layer_idx",
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


def _extract_value(result: Any, var_name: str, pos_ret: Optional[int]) -> Any:
    """
    Extract the target value from the decorated function's return payload.
    """
    if pos_ret is not None:
        try:
            return result[pos_ret]
        except Exception as exc:
            raise ValueError(
                f"visTvis_store could not read position {pos_ret} from return value."
            ) from exc

    if isinstance(result, dict) and var_name in result:
        return result[var_name]

    if hasattr(result, var_name):
        return getattr(result, var_name)

    raise ValueError(
        "visTvis_store could not find the value to store. Provide pos_ret or ensure "
        f"the returned object exposes `{var_name}`."
    )


__all__ = ["visTvis_store", "visTvis_layer_counter", "LayerCounter"]
