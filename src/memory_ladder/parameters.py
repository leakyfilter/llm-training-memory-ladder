"""Approximate decoder-only Transformer parameter-count formulas."""

from __future__ import annotations

from memory_ladder.config import TransformerConfig


def token_embedding_parameters(config: TransformerConfig) -> int:
    # Assumption: one learned embedding vector of size H for each vocabulary row.
    return config.vocab_size * config.hidden_size


def attention_projection_parameters_per_layer(config: TransformerConfig) -> int:
    hidden_size = config.hidden_size
    # Assumption: dense Q, K, V, and output projections, each with H * H weights.
    return 4 * hidden_size * hidden_size


def swiglu_mlp_parameters_per_layer(config: TransformerConfig) -> int:
    hidden_size = config.hidden_size
    intermediate_size = int(config.mlp_expansion_ratio * hidden_size)
    # Assumption: SwiGLU has gate, up, and down projections with no bias terms.
    return 3 * hidden_size * intermediate_size


def transformer_block_parameters_per_layer(config: TransformerConfig) -> int:
    # Assumption: the v0 decoder block parameter count is the attention
    # projection weights plus SwiGLU MLP weights. Norms are modeled separately
    # by norm_parameters() so shared block math does not double-count them.
    return (
        attention_projection_parameters_per_layer(config)
        + swiglu_mlp_parameters_per_layer(config)
    )


def norm_parameters(config: TransformerConfig) -> int:
    if not config.include_norm_parameters:
        return 0

    hidden_size = config.hidden_size
    # Assumption: pre-attention and pre-MLP norms per block, plus one final norm.
    # Each norm has one learned scale vector of length H and no bias term.
    return (2 * config.num_layers * hidden_size) + hidden_size


def lm_head_parameters(config: TransformerConfig) -> int:
    if config.tie_lm_head:
        return 0

    # Assumption: untied LM head is a separate vocab_size * H projection matrix.
    return config.vocab_size * config.hidden_size


def total_parameter_count(config: TransformerConfig) -> int:
    """Return total model parameters under the v0 decoder-only approximation."""

    return (
        token_embedding_parameters(config)
        + (config.num_layers * transformer_block_parameters_per_layer(config))
        + norm_parameters(config)
        + lm_head_parameters(config)
    )
