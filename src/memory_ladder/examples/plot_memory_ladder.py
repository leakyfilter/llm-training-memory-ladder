"""Create the v0 memory ladder stacked-bar plot."""

from __future__ import annotations

from pathlib import Path

from memory_ladder.examples.defaults import default_memory_breakdowns
from memory_ladder.plotting import plot_memory_by_strategy


DEFAULT_OUTPUT_PATH = Path("outputs") / "memory_by_strategy.png"


def main() -> None:
    output_path = plot_memory_by_strategy(
        default_memory_breakdowns(),
        DEFAULT_OUTPUT_PATH,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
