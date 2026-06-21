"""Dataclasses shared by memory estimators.

All memory quantities in this project are represented in bytes unless a field
or property explicitly includes ``gb`` in its name.
"""

from __future__ import annotations

from dataclasses import dataclass, field


BYTES_PER_GB = 1_000_000_000


@dataclass(frozen=True)
class TransformerConfig:
    """Decoder-only Transformer shape assumptions."""

    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    vocab_size: int = 50_257
    mlp_expansion_ratio: float = 4.0
    tie_lm_head: bool = True
    include_norm_parameters: bool = True

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.mlp_expansion_ratio <= 0:
            raise ValueError("mlp_expansion_ratio must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    """Training-memory assumptions for one memory estimate."""

    sequence_length: int = 1024
    microbatch_size: int = 1
    dp_size: int = 1
    dtype_bytes: int = 2
    grad_bytes: int = 2
    optimizer_state_bytes_per_param: int = 8
    activation_multiplier_per_layer: float = 6.0
    zero3_gather_buffer_override_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if self.dp_size <= 0:
            raise ValueError("dp_size must be positive")
        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")
        if self.grad_bytes <= 0:
            raise ValueError("grad_bytes must be positive")
        if self.optimizer_state_bytes_per_param < 0:
            raise ValueError("optimizer_state_bytes_per_param must be non-negative")
        if self.activation_multiplier_per_layer < 0:
            raise ValueError("activation_multiplier_per_layer must be non-negative")
        if (
            self.zero3_gather_buffer_override_bytes is not None
            and self.zero3_gather_buffer_override_bytes < 0
        ):
            raise ValueError("zero3_gather_buffer_override_bytes must be non-negative")


@dataclass(frozen=True)
class MemoryBreakdown:
    """Per-GPU memory estimate for a strategy.

    The component fields are bytes. ``total_memory_bytes`` is computed from the
    component fields to keep totals consistent.
    """

    strategy: str
    world_size: int
    dp_size: int
    parameter_memory_bytes: int
    gradient_memory_bytes: int
    optimizer_memory_bytes: int
    activation_memory_bytes: int
    temporary_memory_bytes: int
    key_assumption: str
    total_memory_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "world_size",
            "dp_size",
            "parameter_memory_bytes",
            "gradient_memory_bytes",
            "optimizer_memory_bytes",
            "activation_memory_bytes",
            "temporary_memory_bytes",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

        total_memory_bytes = (
            self.parameter_memory_bytes
            + self.gradient_memory_bytes
            + self.optimizer_memory_bytes
            + self.activation_memory_bytes
            + self.temporary_memory_bytes
        )
        object.__setattr__(self, "total_memory_bytes", total_memory_bytes)

    @property
    def total_memory_gb(self) -> float:
        """Human-readable decimal GB total."""

        return self.total_memory_bytes / BYTES_PER_GB

    def to_row(self) -> dict[str, int | float | str]:
        """Return a pandas-friendly row with byte and decimal-GB quantities."""

        return {
            "strategy": self.strategy,
            "world_size": self.world_size,
            "dp_size": self.dp_size,
            "parameter_memory_bytes": self.parameter_memory_bytes,
            "gradient_memory_bytes": self.gradient_memory_bytes,
            "optimizer_memory_bytes": self.optimizer_memory_bytes,
            "activation_memory_bytes": self.activation_memory_bytes,
            "temporary_memory_bytes": self.temporary_memory_bytes,
            "total_memory_bytes": self.total_memory_bytes,
            "parameter_memory_gb": self.parameter_memory_bytes / BYTES_PER_GB,
            "gradient_memory_gb": self.gradient_memory_bytes / BYTES_PER_GB,
            "optimizer_memory_gb": self.optimizer_memory_bytes / BYTES_PER_GB,
            "activation_memory_gb": self.activation_memory_bytes / BYTES_PER_GB,
            "temporary_memory_gb": self.temporary_memory_bytes / BYTES_PER_GB,
            "total_memory_gb": self.total_memory_gb,
            "key_assumption": self.key_assumption,
        }
