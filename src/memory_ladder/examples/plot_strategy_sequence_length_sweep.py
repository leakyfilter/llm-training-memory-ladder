"""Plot stacked memory components for one strategy across sequence lengths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from memory_ladder.examples.sweep_sequence_length import build_sequence_length_sweep
from memory_ladder.plotting import plot_stacked_memory_by_sequence_length


DEFAULT_STRATEGY = "ZeRO-3"


def strategy_slug(strategy: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", strategy.lower()).strip("_")
    return slug or "strategy"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help="Strategy name to plot, for example ZeRO-3 or DDP.",
    )
    parser.add_argument(
        "--gpu-memory-budget-gb",
        type=float,
        default=24.0,
        help="GPU memory budget in decimal GB used for the underlying sweep CSV rows.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Path for the PNG output.",
    )
    args = parser.parse_args(argv)

    output_png = args.output_png
    if output_png is None:
        output_png = (
            Path("outputs")
            / f"{strategy_slug(args.strategy)}_sequence_length_stacked.png"
        )

    table = build_sequence_length_sweep(args.gpu_memory_budget_gb)
    output_path = plot_stacked_memory_by_sequence_length(
        table.to_dict("records"),
        output_png,
        strategy=args.strategy,
        gpu_memory_budget_gb=args.gpu_memory_budget_gb,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
