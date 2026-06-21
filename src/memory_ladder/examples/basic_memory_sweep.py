"""Print a small strategy comparison table."""

from __future__ import annotations

import pandas as pd

from memory_ladder import DDP, DenseSingleGPU, TrainingConfig, TransformerConfig
from memory_ladder import ZeRO1, ZeRO2, ZeRO3


def build_table() -> pd.DataFrame:
    model_config = TransformerConfig(
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        vocab_size=32_000,
        mlp_expansion_ratio=4.0,
        tie_lm_head=True,
    )
    training_config = TrainingConfig(
        sequence_length=1024,
        microbatch_size=1,
        dp_size=4,
        dtype_bytes=2,
        grad_bytes=2,
        optimizer_state_bytes_per_param=8,
        activation_multiplier_per_layer=6.0,
    )

    strategies = [DenseSingleGPU(), DDP(), ZeRO1(), ZeRO2(), ZeRO3()]
    rows = [
        strategy.estimate(model_config, training_config).to_row()
        for strategy in strategies
    ]
    columns = [
        "strategy",
        "world_size",
        "dp_size",
        "parameter_memory_gb",
        "gradient_memory_gb",
        "optimizer_memory_gb",
        "activation_memory_gb",
        "temporary_memory_gb",
        "total_memory_gb",
        "key_assumption",
    ]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    table = build_table()
    print(table.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
