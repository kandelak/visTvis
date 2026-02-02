from __future__ import annotations

import os
from typing import Iterable, List, Union

import matplotlib.pyplot as plt
from PIL import Image

from .models import OverlayOutputPath

HeadLabel = Union[int, str]


def _discover_queries(output: OverlayOutputPath, layers: List[int], heads: List[HeadLabel]) -> List[int]:
    """Find query indices by inspecting overlay filenames in the first available head."""
    for layer_id in layers:
        for head in heads:
            head_dir = output.head_dir(layer_id, head)
            overlay_dir = os.path.join(head_dir, "overlays")
            if not os.path.isdir(overlay_dir):
                continue
            queries = []
            for name in os.listdir(overlay_dir):
                if name.startswith("overlay_") and name.endswith(".png"):
                    stem = name[len("overlay_") : -len(".png")]
                    if stem.isdigit():
                        queries.append(int(stem))
            if queries:
                return sorted(queries)
    return []


def plot_attention_for_reconstruction(
    output_path: Union[str, OverlayOutputPath],
    *,
    dpi: int = 300,
) -> None:
    """Create a PDF per query stacking layers as rows and heads as columns."""
    output = output_path if isinstance(output_path, OverlayOutputPath) else OverlayOutputPath(output_path=output_path)
    layers = output.list_layers()
    if not layers:
        raise ValueError(f"No layers found under {output.output_path}")

    # Assume heads consistent across layers; use the first layer to infer ordering.
    heads = output.list_heads(layers[0])
    if not heads:
        raise ValueError(f"No heads found under {output.layer_dir(layers[0])}")

    queries = _discover_queries(output, layers, heads)
    if not queries:
        raise ValueError("No overlays found to plot.")

    for q in queries:
        fig, axes = plt.subplots(len(layers), len(heads), figsize=(3 * len(heads), 3 * len(layers)))
        if len(layers) == 1 and len(heads) == 1:
            axes = [[axes]]
        elif len(layers) == 1:
            axes = [axes]
        elif len(heads) == 1:
            axes = [[ax] for ax in axes]

        for i, layer_id in enumerate(layers):
            for j, head in enumerate(heads):
                ax = axes[i][j]
                overlay_file = output.overlay_file(layer_id, head, q)
                if os.path.exists(overlay_file):
                    img = Image.open(overlay_file)
                    ax.imshow(img)
                ax.axis("off")

        pdf_path = os.path.join(output.output_path, f"attention_query_{q}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)


__all__ = ["plot_attention_for_reconstruction"]
