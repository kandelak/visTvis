from __future__ import annotations

import os
from typing import Dict, List, Literal, Tuple, Union

from tqdm import tqdm

from .models import AttentionPath
from .overlay_attention import (
    HeadSpec,
    QueryIdxSpec,
    _load_array,
    _select_queries_interactively,
    overlay_query_to_frame_attention,
)
from .static_metadata import StaticMetadata

HeadSelection = Union[int, Literal["mean", "all"]]
ResultKey = Tuple[int, HeadSpec]
_ENV_METADATA = "VIS_TVIS_METADATA_PATH"


def _resolve_heads(attn_path: str, head: HeadSelection) -> List[HeadSpec]:
    if head == "all":
        attn = _load_array(attn_path)
        if attn.ndim != 4:
            raise ValueError(f"Expected attention tensor with 4 dims; got {attn.shape}")
        _, head_count, _, _ = attn.shape
        return list(range(head_count)) + ["mean"]
    if head == "mean" or isinstance(head, int):
        return [head]
    raise ValueError(f"Unsupported head value: {head!r}")


def _resolve_metadata(metadata: StaticMetadata | None) -> StaticMetadata:
    if metadata:
        return metadata
    path = os.getenv(_ENV_METADATA)
    if not path:
        raise ValueError(
            f"metadata not provided and environment variable {_ENV_METADATA} is not set."
        )
    return StaticMetadata.from_yaml_file(path)


def _resolve_query_indices(
    query_idx: QueryIdxSpec,
    *,
    input_path: str,
    metadata: StaticMetadata,
    batch_idx: int,
    query_frame_idx: int,
) -> List[int] | str:
    """Interactive selection is performed once and shared across layers."""
    if not (isinstance(query_idx, str) and query_idx == "interactive"):
        return query_idx

    x = _load_array(input_path)  # (B, N_fr, C, H, W)
    if x.ndim != 5:
        raise ValueError(f"Expected input to have 5 dims (B, N_fr, C, H, W); got {x.shape}")
    _, N_fr, _, H, W = x.shape
    if not (0 <= query_frame_idx < N_fr):
        raise ValueError(f"query_frame_idx out of range: {query_frame_idx} (N_fr={N_fr})")

    patch_size = int(getattr(metadata, "patch_size"))
    num_special = int(getattr(metadata, "num_special_tokens"))
    if patch_size <= 0:
        raise ValueError(f"metadata.patch_size must be > 0; got {patch_size}")

    if (H % patch_size) != 0 or (W % patch_size) != 0:
        raise ValueError(
            f"Input frame size (H={H}, W={W}) must be divisible by patch_size={patch_size}."
        )
    grid_h = H // patch_size
    grid_w = W // patch_size

    frame_for_selection = x[batch_idx, query_frame_idx]
    queries = _select_queries_interactively(
        frame_for_selection,
        grid_w=grid_w,
        grid_h=grid_h,
        patch_size=patch_size,
        num_special=num_special,
    )
    if not queries:
        raise ValueError("No queries selected in interactive mode.")
    return queries


def run_overlay_for_reconstruction(
    paths: AttentionPath,
    *,
    query_frame_idx: int,
    query_idx: QueryIdxSpec,
    key_frame_idx: int,
    head: HeadSelection,
    output_path: str,
    metadata: StaticMetadata | None = None,
    is_cross_attention: bool = False,
    batch_idx: int = 0,
    alpha: float = 0.45,
    cmap_name: str = "jet",
    eps: float = 1e-8,
) -> Dict[ResultKey, List[str]]:
    """Run overlays for each available layer; returns mapping of (layer_id, head) -> saved paths."""
    layer_ids = paths.list_layers()
    if not layer_ids:
        raise ValueError(f"No attention layers found under {paths.attn_root}")

    metadata = _resolve_metadata(metadata)
    input_path = paths.input_layer_file(0)
    query_idx = _resolve_query_indices(
        query_idx,
        input_path=input_path,
        metadata=metadata,
        batch_idx=batch_idx,
        query_frame_idx=query_frame_idx,
    )

    results: Dict[ResultKey, List[str]] = {}

    global_query_dir = os.path.join(output_path, "query_frames")

    for layer_id in tqdm(layer_ids, desc="Layers"):
        attn_path = paths.attn_layer_file(layer_id)
        heads = _resolve_heads(attn_path, head)
        for h in tqdm(heads, desc=f"Layer {layer_id} heads", leave=False):
            head_label = str(h)
            layer_output = os.path.join(
                output_path,
                f"layer=layer_{layer_id}",
                f"head={head_label}",
            )
            saved = overlay_query_to_frame_attention(
                attn_path=attn_path,
                input_path=input_path,
                metadata=metadata,
                query_frame_idx=query_frame_idx,
                query_idx=query_idx,
                key_frame_idx=key_frame_idx,
                head=h,
                output_path=layer_output,
                batch_idx=batch_idx,
                alpha=alpha,
                cmap_name=cmap_name,
                eps=eps,
                is_cross_attention=is_cross_attention,
                query_output_path=global_query_dir,
            )
            results[(layer_id, h)] = saved

    return results


__all__ = ["run_overlay_for_reconstruction", "AttentionPath", "HeadSelection"]
