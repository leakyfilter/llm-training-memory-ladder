"""Print a small strategy comparison table."""

from __future__ import annotations

import pandas as pd

from memory_ladder.examples.defaults import default_memory_breakdowns


def build_table() -> pd.DataFrame:
    rows = [breakdown.to_row() for breakdown in default_memory_breakdowns()]
    columns = [
        "strategy",
        "world_size",
        "dp_size",
        "activation_checkpointing",
        "parameter_memory_gb",
        "gradient_memory_gb",
        "optimizer_memory_gb",
        "activation_memory_gb",
        "temporary_memory_gb",
        "total_memory_gb",
        "key_assumption",
        "recompute_note",
    ]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    table = build_table()
    print(table.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
