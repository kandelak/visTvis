"""
Lightweight storage helpers built on top of torch.save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency issue
    raise ImportError(
        "torch is required for visTvis storage. Install with `pip install torch`."
    ) from exc


def store_value(
        value: torch.Tensor,
        base_folder_path: str,
        identifier: str,
        layer: int,
        var_name: str,
        folder_name: str,
) -> Optional[Path]:
    """
    Persist value to a structured directory layout:
    base_folder_path/identifier/folder_name/layer_{layer}/{var_name}.pt
    """
    if value is None:
        return None

    root = Path(base_folder_path).expanduser()
    target_dir = root / identifier / folder_name / f"layer_{layer}"
    target_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(value, torch.Tensor):
        target_path = target_dir / f"{var_name}.pt"
        torch.save(value, target_path)
    else:
        raise TypeError(
            f"Unsupported value type for storage: {type(value)}. "
            "Supported types are torch.Tensor and Metadata."
        )
    return target_path


__all__ = ["store_value"]
