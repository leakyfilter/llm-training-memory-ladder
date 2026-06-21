"""Matplotlib plotting helpers for memory estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_ladder.config import BYTES_PER_GB, MemoryBreakdown


COMPONENTS = [
    ("parameter_memory_bytes", "parameters", "#4C78A8"),
    ("gradient_memory_bytes", "gradients", "#F58518"),
    ("optimizer_memory_bytes", "optimizer states", "#54A24B"),
    ("activation_memory_bytes", "activations", "#B279A2"),
    ("temporary_memory_bytes", "temporary/gather buffers", "#E45756"),
]


def plot_memory_by_strategy(
    breakdowns: Sequence[MemoryBreakdown], output_path: str | Path
) -> Path:
    """Write a stacked-bar memory plot and return the output path."""

    if not breakdowns:
        raise ValueError("breakdowns must not be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategy_names = [breakdown.strategy for breakdown in breakdowns]
    bottom_gb = [0.0 for _ in breakdowns]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for bytes_field, label, color in COMPONENTS:
        # Assumption: plotted component heights use decimal GB, matching
        # MemoryBreakdown.to_row() and total_memory_gb.
        values_gb = [
            getattr(breakdown, bytes_field) / BYTES_PER_GB
            for breakdown in breakdowns
        ]
        ax.bar(
            strategy_names,
            values_gb,
            bottom=bottom_gb,
            label=label,
            color=color,
            width=0.68,
        )
        bottom_gb = [
            previous_gb + value_gb
            for previous_gb, value_gb in zip(bottom_gb, values_gb, strict=True)
        ]

    ax.set_title("Per-GPU Training Memory by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Memory (GB)")
    ax.set_ylim(0, max(bottom_gb) * 1.12)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path
