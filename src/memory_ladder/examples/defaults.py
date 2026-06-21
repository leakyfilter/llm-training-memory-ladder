"""Shared defaults for example scripts."""

from __future__ import annotations

from memory_ladder import DDP, DenseSingleGPU, TrainingConfig, TransformerConfig
from memory_ladder import ZeRO1, ZeRO2, ZeRO3
from memory_ladder.config import MemoryBreakdown


def default_model_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        vocab_size=32_000,
        mlp_expansion_ratio=4.0,
        tie_lm_head=True,
    )


def default_training_config() -> TrainingConfig:
    return TrainingConfig(
        sequence_length=1024,
        microbatch_size=1,
        dp_size=4,
        dtype_bytes=2,
        grad_bytes=2,
        optimizer_state_bytes_per_param=8,
        activation_multiplier_per_layer=6.0,
    )


def default_memory_breakdowns() -> list[MemoryBreakdown]:
    model_config = default_model_config()
    training_config = default_training_config()
    strategies = [DenseSingleGPU(), DDP(), ZeRO1(), ZeRO2(), ZeRO3()]
    return [
        strategy.estimate(model_config, training_config)
        for strategy in strategies
    ]
