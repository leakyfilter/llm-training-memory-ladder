"""Shape-sweep utilities for comparing memory strategies."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Iterable, Sequence

from memory_ladder.config import BYTES_PER_GB, MemoryBreakdown
from memory_ladder.config import TrainingConfig, TransformerConfig
from memory_ladder.estimators import DDP, DenseSingleGPU, MemoryEstimator
from memory_ladder.estimators import ZeRO1, ZeRO2, ZeRO3


def default_strategy_estimators() -> list[MemoryEstimator]:
    return [DenseSingleGPU(), DDP(), ZeRO1(), ZeRO2(), ZeRO3()]


def estimate_strategy_breakdowns(
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    strategies: Sequence[MemoryEstimator] | None = None,
) -> list[MemoryBreakdown]:
    estimators = list(strategies) if strategies is not None else default_strategy_estimators()
    return [
        estimator.estimate(model_config, training_config)
        for estimator in estimators
    ]


def sweep_shapes(
    *,
    base_model_config: TransformerConfig,
    base_training_config: TrainingConfig,
    hidden_sizes: Iterable[int],
    num_layers_values: Iterable[int],
    sequence_lengths: Iterable[int],
    microbatch_sizes: Iterable[int],
    gpu_memory_budget_gb: float = 24.0,
    strategies: Sequence[MemoryEstimator] | None = None,
) -> list[dict[str, int | float | str | bool]]:
    """Estimate strategy memory for each requested model/training shape."""

    if gpu_memory_budget_gb <= 0:
        raise ValueError("gpu_memory_budget_gb must be positive")

    rows: list[dict[str, int | float | str | bool]] = []
    # Assumption: GPU budget uses decimal GB, matching MemoryBreakdown.total_memory_gb.
    budget_memory_bytes = int(gpu_memory_budget_gb * BYTES_PER_GB)

    for hidden_size, num_layers, sequence_length, microbatch_size in product(
        hidden_sizes,
        num_layers_values,
        sequence_lengths,
        microbatch_sizes,
    ):
        model_config = replace(
            base_model_config,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        training_config = replace(
            base_training_config,
            sequence_length=sequence_length,
            microbatch_size=microbatch_size,
        )
        breakdowns = estimate_strategy_breakdowns(
            model_config,
            training_config,
            strategies=strategies,
        )
        winning_breakdown = min(
            breakdowns,
            key=lambda breakdown: breakdown.total_memory_bytes,
        )

        for breakdown in breakdowns:
            row = breakdown.to_row()
            row.update(
                {
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "sequence_length": sequence_length,
                    "microbatch_size": microbatch_size,
                    "gpu_memory_budget_gb": gpu_memory_budget_gb,
                    "oom": breakdown.total_memory_bytes > budget_memory_bytes,
                    "is_winner": breakdown.strategy == winning_breakdown.strategy,
                    "winning_strategy": winning_breakdown.strategy,
                }
            )
            rows.append(row)

    return rows


def sweep_sequence_lengths(
    *,
    base_model_config: TransformerConfig,
    base_training_config: TrainingConfig,
    sequence_lengths: Iterable[int],
    gpu_memory_budget_gb: float = 24.0,
    strategies: Sequence[MemoryEstimator] | None = None,
) -> list[dict[str, int | float | str | bool]]:
    """Sweep only sequence length while holding model shape and microbatch fixed."""

    return sweep_shapes(
        base_model_config=base_model_config,
        base_training_config=base_training_config,
        hidden_sizes=[base_model_config.hidden_size],
        num_layers_values=[base_model_config.num_layers],
        sequence_lengths=sequence_lengths,
        microbatch_sizes=[base_training_config.microbatch_size],
        gpu_memory_budget_gb=gpu_memory_budget_gb,
        strategies=strategies,
    )
