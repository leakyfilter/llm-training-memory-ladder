"""Sweep sequence length and plot per-GPU memory by strategy."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from memory_ladder.examples.defaults import default_model_config
from memory_ladder.examples.defaults import default_training_config
from memory_ladder.plotting import plot_sequence_length_sweep
from memory_ladder.sweeps import sweep_sequence_lengths


DEFAULT_OUTPUT_CSV = Path("outputs") / "sequence_length_sweep.csv"
DEFAULT_OUTPUT_PNG = Path("outputs") / "sequence_length_sweep.png"
DEFAULT_SEQUENCE_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]


def build_sequence_length_sweep(
    gpu_memory_budget_gb: float = 24.0,
) -> pd.DataFrame:
    model_config = default_model_config()
    # Assumption: the sequence-length example keeps the Llama-70B-scale model
    # shape but uses a larger data-parallel group so the 24 GB budget line shows
    # where activation growth pushes even ZeRO-3 over budget.
    training_config = replace(default_training_config(), dp_size=64)
    rows = sweep_sequence_lengths(
        base_model_config=model_config,
        base_training_config=training_config,
        sequence_lengths=DEFAULT_SEQUENCE_LENGTHS,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
    )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-memory-budget-gb",
        type=float,
        default=24.0,
        help="GPU memory budget in decimal GB.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for the CSV sweep output.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="Path for the PNG sweep plot.",
    )
    args = parser.parse_args(argv)

    table = build_sequence_length_sweep(args.gpu_memory_budget_gb)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    plot_sequence_length_sweep(
        table.to_dict("records"),
        args.output_png,
        gpu_memory_budget_gb=args.gpu_memory_budget_gb,
    )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
