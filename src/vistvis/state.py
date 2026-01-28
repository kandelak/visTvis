"""
State utilities for tracking layers across decorated functions.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Dict


class LayerState:
    """
    Stores per-identifier counters and exposes a context-aware view of the
    current layer. The ContextVar keeps nested calls isolated while the sticky
    mapping remembers the last value seen for each identifier.
    """

    def __init__(self) -> None:
        self._current_layers: ContextVar[Dict[str, int]] = ContextVar(
            "vistvis_current_layers", default={}
        )
        self._sticky_layers: Dict[str, int] = {}

    def get_layer(self, identifier: str) -> int:
        layers = self._current_layers.get()
        if identifier in layers:
            return layers[identifier]
        return self._sticky_layers.get(identifier, 0)

    def push(self, identifier: str, value: int):
        """
        Push the active layer for an identifier. Returns a token that must be
        passed back into pop() to restore the previous context.
        """
        layers = dict(self._current_layers.get())
        layers[identifier] = int(value)
        token = self._current_layers.set(layers)
        self._sticky_layers[identifier] = int(value)
        return token

    def pop(self, token) -> None:
        self._current_layers.reset(token)

    def update(self, identifier: str, value: int) -> int:
        """
        Update the current layer value for the identifier and remember it as
        the latest value for the next call.
        """
        layers = dict(self._current_layers.get())
        layers[identifier] = int(value)
        self._current_layers.set(layers)
        self._sticky_layers[identifier] = int(value)
        return int(value)

    def bump(self, identifier: str) -> int:
        """
        Increment the layer counter for an identifier and return the new value.
        """
        return self.update(identifier, self.get_layer(identifier) + 1)


state = LayerState()

__all__ = ["state", "LayerState"]
