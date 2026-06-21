"""Matplotlib plotting helpers for memory estimates."""

from __future__ import annotations

from collections.abc import Mapping
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


def plot_sequence_length_sweep(
    rows: Sequence[Mapping[str, int | float | str | bool]],
    output_path: str | Path,
    gpu_memory_budget_gb: float = 24.0,
) -> Path:
    """Write a line plot of total per-GPU memory versus sequence length."""

    if not rows:
        raise ValueError("rows must not be empty")
    if gpu_memory_budget_gb <= 0:
        raise ValueError("gpu_memory_budget_gb must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategies = list(dict.fromkeys(str(row["strategy"]) for row in rows))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for strategy in strategies:
        strategy_rows = [
            row
            for row in rows
            if row["strategy"] == strategy
        ]
        strategy_rows.sort(key=lambda row: int(row["sequence_length"]))
        x_values = [int(row["sequence_length"]) for row in strategy_rows]
        y_values = [float(row["total_memory_gb"]) for row in strategy_rows]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=strategy)

    ax.axhline(
        gpu_memory_budget_gb,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label=f"{gpu_memory_budget_gb:g} GB budget",
    )
    ax.set_title("Per-GPU Memory Versus Sequence Length")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Total memory (GB)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def plot_stacked_memory_by_sequence_length(
    rows: Sequence[Mapping[str, int | float | str | bool]],
    output_path: str | Path,
    strategy: str,
    gpu_memory_budget_gb: float = 24.0,
) -> Path:
    """Write stacked component bars across sequence lengths for one strategy."""

    if gpu_memory_budget_gb <= 0:
        raise ValueError("gpu_memory_budget_gb must be positive")

    strategy_rows = [
        row
        for row in rows
        if str(row["strategy"]) == strategy
    ]
    if not strategy_rows:
        raise ValueError(f"no rows found for strategy {strategy!r}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategy_rows.sort(key=lambda row: int(row["sequence_length"]))
    sequence_labels = [
        str(row["sequence_length"])
        for row in strategy_rows
    ]
    bottom_gb = [0.0 for _ in strategy_rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for bytes_field, label, color in COMPONENTS:
        # Assumption: stacked sequence bars use decimal GB, matching
        # MemoryBreakdown.to_row() and the other v0 plot outputs.
        values_gb = [
            int(row[bytes_field]) / BYTES_PER_GB
            for row in strategy_rows
        ]
        ax.bar(
            sequence_labels,
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

    ax.set_title(f"{strategy} Per-GPU Memory by Sequence Length")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Memory (GB)")
    ax.set_ylim(0, max(bottom_gb) * 1.12)
    ax.axhline(
        gpu_memory_budget_gb,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label=f"{gpu_memory_budget_gb:g} GB budget",
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path
