"""Shared defaults for example scripts."""

from __future__ import annotations

from dataclasses import replace

from memory_ladder import DDP, DenseSingleGPU, TrainingConfig, TransformerConfig
from memory_ladder import ZeRO1, ZeRO2, ZeRO3
from memory_ladder.config import MemoryBreakdown


def default_model_config() -> TransformerConfig:
    # Assumption: this is a Llama-70B-scale shape under the v0 decoder formula.
    # v0 models full Q/K/V/O attention and does not yet model GQA, so the MLP
    # expansion ratio is set to 3.0 to keep total parameters close to 70B.
    return TransformerConfig(
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        vocab_size=32_000,
        mlp_expansion_ratio=3.0,
        tie_lm_head=True,
    )


def default_training_config() -> TrainingConfig:
    # Assumption: use a long context in examples so activation memory is large
    # enough to make checkpointing and ZeRO tradeoffs visible in stacked plots.
    return TrainingConfig(
        sequence_length=32768,
        microbatch_size=1,
        dp_size=8,
        dtype_bytes=2,
        grad_bytes=2,
        optimizer_state_bytes_per_param=8,
        activation_multiplier_per_layer=6.0,
    )


def default_strategy_estimators() -> list[DenseSingleGPU | DDP | ZeRO1 | ZeRO2 | ZeRO3]:
    return [DenseSingleGPU(), DDP(), ZeRO1(), ZeRO2(), ZeRO3()]


def default_memory_breakdowns(
    compare_activation_checkpointing: bool = True,
) -> list[MemoryBreakdown]:
    model_config = default_model_config()
    training_config = default_training_config()
    strategies = default_strategy_estimators()

    if not compare_activation_checkpointing:
        return [
            strategy.estimate(model_config, training_config)
            for strategy in strategies
        ]

    checkpointing_configs = [
        training_config,
        replace(training_config, activation_checkpointing=True),
    ]
    return [
        strategy.estimate(model_config, checkpointing_config)
        for checkpointing_config in checkpointing_configs
        for strategy in strategies
    ]
