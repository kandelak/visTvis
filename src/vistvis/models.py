from __future__ import annotations

import os
import re
from typing import List, Union

from pydantic import BaseModel, field_validator


class AttentionPath(BaseModel):
    """Path helper for saved attention/input artifacts."""

    base_dir: str
    experiment_name: str
    attn_filename: str = "attn.pt"
    input_filename: str = "input.pt"

    @field_validator("base_dir")
    @classmethod
    def _expand_base_dir(cls, v: str) -> str:
        return os.path.expanduser(v)

    @property
    def experiment_root(self) -> str:
        return os.path.join(self.base_dir, self.experiment_name)

    @property
    def attn_root(self) -> str:
        return os.path.join(self.experiment_root, "attn")

    @property
    def input_root(self) -> str:
        return os.path.join(self.experiment_root, "input")

    def attn_layer_file(self, layer_id: int) -> str:
        return os.path.join(self.attn_root, f"layer_{layer_id}", self.attn_filename)

    def input_layer_file(self, layer_id: int = 0) -> str:
        return os.path.join(self.input_root, f"layer_{layer_id}", self.input_filename)

    def list_layers(self) -> List[int]:
        """Return sorted layer ids discovered under attn_root."""
        if not os.path.isdir(self.attn_root):
            return []
        pattern = re.compile(r"layer_(\d+)$")
        layers: List[int] = []
        for name in os.listdir(self.attn_root):
            match = pattern.match(name)
            if match:
                layers.append(int(match.group(1)))
        return sorted(layers)


__all__ = ["AttentionPath"]


class OverlayOutputPath(BaseModel):
    """Path helper for overlay outputs."""

    output_path: str

    @field_validator("output_path")
    @classmethod
    def _expand_output(cls, v: str) -> str:
        return os.path.expanduser(v)

    @property
    def query_frames_dir(self) -> str:
        return os.path.join(self.output_path, "query_frames")

    def layer_dir(self, layer_id: int) -> str:
        return os.path.join(self.output_path, f"layer=layer_{layer_id}")

    def head_dir(self, layer_id: int, head: Union[int, str]) -> str:
        return os.path.join(self.layer_dir(layer_id), f"head={head}")

    def overlay_file(self, layer_id: int, head: Union[int, str], query_idx: int) -> str:
        return os.path.join(self.head_dir(layer_id, head), "overlays", f"overlay_{query_idx}.png")

    def list_layers(self) -> List[int]:
        if not os.path.isdir(self.output_path):
            return []
        pattern = re.compile(r"layer=layer_(\d+)$")
        layers: List[int] = []
        for name in os.listdir(self.output_path):
            m = pattern.match(name)
            if m:
                layers.append(int(m.group(1)))
        return sorted(layers)

    def list_heads(self, layer_id: int) -> List[Union[int, str]]:
        layer_path = self.layer_dir(layer_id)
        if not os.path.isdir(layer_path):
            return []
        pattern = re.compile(r"head=(.+)$")
        heads: List[Union[int, str]] = []
        for name in os.listdir(layer_path):
            m = pattern.match(name)
            if not m:
                continue
            value = m.group(1)
            if value.isdigit():
                heads.append(int(value))
            else:
                heads.append(value)
        # Sort numeric heads first, then others.
        nums = sorted(h for h in heads if isinstance(h, int))
        non_nums = sorted(str(h) for h in heads if not isinstance(h, int))
        return nums + non_nums


__all__.extend(["OverlayOutputPath"])
