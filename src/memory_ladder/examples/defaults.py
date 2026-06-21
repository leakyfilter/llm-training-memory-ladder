"""Shared defaults for example scripts."""

from __future__ import annotations

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
    return TrainingConfig(
        sequence_length=4096,
        microbatch_size=1,
        dp_size=8,
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
