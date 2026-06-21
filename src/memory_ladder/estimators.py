"""Per-GPU memory estimators for the v0 parallelism ladder."""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from typing import Protocol

from memory_ladder.config import MemoryBreakdown, TrainingConfig, TransformerConfig
from memory_ladder.parameters import (
    total_parameter_count,
    transformer_block_parameters_per_layer,
)


class MemoryEstimator(Protocol):
    """Protocol for strategy classes that produce per-GPU memory estimates."""

    def estimate(
        self, model_config: TransformerConfig, training_config: TrainingConfig
    ) -> MemoryBreakdown:
        """Estimate per-GPU memory in bytes."""


def _dense_component_bytes(
    model_config: TransformerConfig, training_config: TrainingConfig
) -> tuple[int, int, int, int]:
    parameter_count = total_parameter_count(model_config)
    parameter_memory_bytes = parameter_count * training_config.dtype_bytes
    gradient_memory_bytes = parameter_count * training_config.grad_bytes
    optimizer_memory_bytes = (
        parameter_count * training_config.optimizer_state_bytes_per_param
    )
    activation_memory_bytes = estimate_activation_memory_bytes(
        model_config, training_config
    )
    return (
        parameter_memory_bytes,
        gradient_memory_bytes,
        optimizer_memory_bytes,
        activation_memory_bytes,
    )


def _shard_bytes(byte_count: int, shard_count: int) -> int:
    # Assumption: shards can be uneven, so per-GPU memory is a conservative ceil.
    return ceil(byte_count / shard_count)


def estimate_activation_memory_bytes(
    model_config: TransformerConfig, training_config: TrainingConfig
) -> int:
    """Estimate activation bytes stored for backward.

    Assumption: activation memory scales with tokens, hidden size, layer count,
    dtype bytes, and a user-controlled multiplier that stands in for attention,
    MLP, residual, dropout, and framework bookkeeping tensors.
    """

    activation_elements = (
        training_config.microbatch_size
        * training_config.sequence_length
        * model_config.hidden_size
        * model_config.num_layers
        * training_config.activation_multiplier_per_layer
    )
    return int(activation_elements * training_config.dtype_bytes)


def estimate_zero3_gather_buffer_bytes(
    model_config: TransformerConfig, training_config: TrainingConfig
) -> int:
    """Estimate ZeRO-3 temporary all-gather memory in bytes."""

    if training_config.zero3_gather_buffer_override_bytes is not None:
        return training_config.zero3_gather_buffer_override_bytes

    # Assumption: ZeRO-3 temporarily materializes one full decoder block's dense
    # projection parameter tensors per GPU during all-gather. The shared block
    # helper owns that parameter-count formula; this function only converts it
    # to bytes for the active parameter dtype.
    return transformer_block_parameters_per_layer(model_config) * training_config.dtype_bytes


class DenseSingleGPU:
    """Dense single-GPU training.

    Replicated tensors: parameters, gradients, optimizer states.
    Sharded tensors: none.
    """

    strategy_name = "DenseSingleGPU"

    def estimate(
        self, model_config: TransformerConfig, training_config: TrainingConfig
    ) -> MemoryBreakdown:
        single_gpu_config = replace(training_config, dp_size=1)
        param_bytes, grad_bytes, optimizer_bytes, activation_bytes = (
            _dense_component_bytes(model_config, single_gpu_config)
        )
        return MemoryBreakdown(
            strategy=self.strategy_name,
            world_size=1,
            dp_size=1,
            parameter_memory_bytes=param_bytes,
            gradient_memory_bytes=grad_bytes,
            optimizer_memory_bytes=optimizer_bytes,
            activation_memory_bytes=activation_bytes,
            temporary_memory_bytes=0,
            key_assumption="Parameters, gradients, and optimizer states are all resident on one GPU.",
        )


class DDP:
    """Data parallel training with fully replicated model state.

    Replicated tensors: parameters, gradients, optimizer states.
    Sharded tensors: none.
    """

    strategy_name = "DDP"

    def estimate(
        self, model_config: TransformerConfig, training_config: TrainingConfig
    ) -> MemoryBreakdown:
        param_bytes, grad_bytes, optimizer_bytes, activation_bytes = (
            _dense_component_bytes(model_config, training_config)
        )
        return MemoryBreakdown(
            strategy=self.strategy_name,
            world_size=training_config.dp_size,
            dp_size=training_config.dp_size,
            parameter_memory_bytes=param_bytes,
            gradient_memory_bytes=grad_bytes,
            optimizer_memory_bytes=optimizer_bytes,
            activation_memory_bytes=activation_bytes,
            temporary_memory_bytes=0,
            key_assumption="DDP replicates parameters, gradients, and optimizer states on every GPU.",
        )


class ZeRO1:
    """ZeRO stage 1 approximation.

    Replicated tensors: parameters, gradients.
    Sharded tensors: optimizer states across the data-parallel group.
    """

    strategy_name = "ZeRO-1"

    def estimate(
        self, model_config: TransformerConfig, training_config: TrainingConfig
    ) -> MemoryBreakdown:
        param_bytes, grad_bytes, optimizer_bytes, activation_bytes = (
            _dense_component_bytes(model_config, training_config)
        )
        return MemoryBreakdown(
            strategy=self.strategy_name,
            world_size=training_config.dp_size,
            dp_size=training_config.dp_size,
            parameter_memory_bytes=param_bytes,
            gradient_memory_bytes=grad_bytes,
            optimizer_memory_bytes=_shard_bytes(
                optimizer_bytes, training_config.dp_size
            ),
            activation_memory_bytes=activation_bytes,
            temporary_memory_bytes=0,
            key_assumption="ZeRO-1 shards optimizer states; parameters and gradients remain replicated.",
        )


class ZeRO2:
    """ZeRO stage 2 approximation.

    Replicated tensors: parameters.
    Sharded tensors: gradients and optimizer states across the data-parallel group.
    """

    strategy_name = "ZeRO-2"

    def estimate(
        self, model_config: TransformerConfig, training_config: TrainingConfig
    ) -> MemoryBreakdown:
        param_bytes, grad_bytes, optimizer_bytes, activation_bytes = (
            _dense_component_bytes(model_config, training_config)
        )
        return MemoryBreakdown(
            strategy=self.strategy_name,
            world_size=training_config.dp_size,
            dp_size=training_config.dp_size,
            parameter_memory_bytes=param_bytes,
            gradient_memory_bytes=_shard_bytes(grad_bytes, training_config.dp_size),
            optimizer_memory_bytes=_shard_bytes(
                optimizer_bytes, training_config.dp_size
            ),
            activation_memory_bytes=activation_bytes,
            temporary_memory_bytes=0,
            key_assumption="ZeRO-2 shards gradients and optimizer states; parameters remain replicated.",
        )


class ZeRO3:
    """ZeRO stage 3 / FSDP-style approximation.

    Replicated tensors: none in steady-state model state.
    Sharded tensors: parameters, gradients, and optimizer states across DP ranks.
    Temporary tensors: estimated gather buffer for materialized parameters.
    """

    strategy_name = "ZeRO-3"

    def estimate(
        self, model_config: TransformerConfig, training_config: TrainingConfig
    ) -> MemoryBreakdown:
        param_bytes, grad_bytes, optimizer_bytes, activation_bytes = (
            _dense_component_bytes(model_config, training_config)
        )
        return MemoryBreakdown(
            strategy=self.strategy_name,
            world_size=training_config.dp_size,
            dp_size=training_config.dp_size,
            parameter_memory_bytes=_shard_bytes(param_bytes, training_config.dp_size),
            gradient_memory_bytes=_shard_bytes(grad_bytes, training_config.dp_size),
            optimizer_memory_bytes=_shard_bytes(
                optimizer_bytes, training_config.dp_size
            ),
            activation_memory_bytes=activation_bytes,
            temporary_memory_bytes=estimate_zero3_gather_buffer_bytes(
                model_config, training_config
            ),
            key_assumption="ZeRO-3 shards parameters, gradients, and optimizer states, plus an estimated one-block gather buffer.",
        )
