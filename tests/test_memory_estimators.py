from memory_ladder import DDP, DenseSingleGPU, TrainingConfig, TransformerConfig
from memory_ladder import ZeRO1, ZeRO2, ZeRO3
from memory_ladder.estimators import (
    estimate_activation_memory_bytes,
    estimate_zero3_gather_buffer_bytes,
)
from memory_ladder.parameters import (
    attention_projection_parameters_per_layer,
    lm_head_parameters,
    norm_parameters,
    swiglu_mlp_parameters_per_layer,
    token_embedding_parameters,
    total_parameter_count,
    transformer_block_parameters_per_layer,
)


def small_model_config(**overrides):
    values = {
        "num_layers": 4,
        "hidden_size": 128,
        "num_attention_heads": 8,
        "vocab_size": 1024,
        "mlp_expansion_ratio": 4.0,
        "tie_lm_head": True,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def training_config(**overrides):
    values = {
        "sequence_length": 128,
        "microbatch_size": 2,
        "dp_size": 4,
        "dtype_bytes": 2,
        "grad_bytes": 2,
        "optimizer_state_bytes_per_param": 8,
        "activation_multiplier_per_layer": 6.0,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def model_state_bytes(breakdown):
    return (
        breakdown.parameter_memory_bytes
        + breakdown.gradient_memory_bytes
        + breakdown.optimizer_memory_bytes
    )


def test_parameter_count_formulas_include_core_decoder_terms():
    config = small_model_config()
    hidden_size = config.hidden_size
    intermediate_size = int(config.mlp_expansion_ratio * hidden_size)

    assert token_embedding_parameters(config) == config.vocab_size * hidden_size
    assert attention_projection_parameters_per_layer(config) == 4 * hidden_size**2
    assert swiglu_mlp_parameters_per_layer(config) == 3 * hidden_size * intermediate_size
    assert transformer_block_parameters_per_layer(config) == (
        attention_projection_parameters_per_layer(config)
        + swiglu_mlp_parameters_per_layer(config)
    )
    assert norm_parameters(config) == (2 * config.num_layers * hidden_size) + hidden_size
    assert lm_head_parameters(config) == 0

    expected_total = (
        config.vocab_size * hidden_size
        + config.num_layers * transformer_block_parameters_per_layer(config)
        + ((2 * config.num_layers * hidden_size) + hidden_size)
    )
    assert total_parameter_count(config) == expected_total


def test_untied_lm_head_adds_projection_parameters():
    tied = small_model_config(tie_lm_head=True)
    untied = small_model_config(tie_lm_head=False)

    assert total_parameter_count(untied) - total_parameter_count(tied) == (
        tied.vocab_size * tied.hidden_size
    )


def test_ddp_does_not_reduce_model_state_memory_per_gpu_versus_single_gpu():
    model_config = small_model_config()
    config = training_config()

    single = DenseSingleGPU().estimate(model_config, config)
    ddp = DDP().estimate(model_config, config)

    assert model_state_bytes(ddp) == model_state_bytes(single)


def test_zero1_reduces_optimizer_memory_by_dp_size():
    model_config = small_model_config()
    config = training_config()

    ddp = DDP().estimate(model_config, config)
    zero1 = ZeRO1().estimate(model_config, config)

    assert zero1.parameter_memory_bytes == ddp.parameter_memory_bytes
    assert zero1.gradient_memory_bytes == ddp.gradient_memory_bytes
    assert zero1.optimizer_memory_bytes == ddp.optimizer_memory_bytes // config.dp_size


def test_zero2_reduces_optimizer_and_gradient_memory_by_dp_size():
    model_config = small_model_config()
    config = training_config()

    ddp = DDP().estimate(model_config, config)
    zero2 = ZeRO2().estimate(model_config, config)

    assert zero2.parameter_memory_bytes == ddp.parameter_memory_bytes
    assert zero2.gradient_memory_bytes == ddp.gradient_memory_bytes // config.dp_size
    assert zero2.optimizer_memory_bytes == ddp.optimizer_memory_bytes // config.dp_size


def test_zero3_shards_model_state_and_adds_estimated_gather_buffer():
    model_config = small_model_config()
    config = training_config()

    ddp = DDP().estimate(model_config, config)
    zero3 = ZeRO3().estimate(model_config, config)
    expected_gather_buffer_bytes = (
        transformer_block_parameters_per_layer(model_config) * config.dtype_bytes
    )

    assert zero3.parameter_memory_bytes == ddp.parameter_memory_bytes // config.dp_size
    assert zero3.gradient_memory_bytes == ddp.gradient_memory_bytes // config.dp_size
    assert zero3.optimizer_memory_bytes == ddp.optimizer_memory_bytes // config.dp_size
    assert zero3.temporary_memory_bytes == expected_gather_buffer_bytes
    assert zero3.total_memory_bytes == (
        zero3.parameter_memory_bytes
        + zero3.gradient_memory_bytes
        + zero3.optimizer_memory_bytes
        + zero3.activation_memory_bytes
        + expected_gather_buffer_bytes
    )


def test_zero3_gather_buffer_allows_explicit_byte_override():
    model_config = small_model_config()
    config = training_config(zero3_gather_buffer_override_bytes=123_456)

    assert estimate_zero3_gather_buffer_bytes(model_config, config) == 123_456
    assert ZeRO3().estimate(model_config, config).temporary_memory_bytes == 123_456


def test_increasing_num_layers_increases_total_memory_monotonically():
    config = training_config()
    small = DenseSingleGPU().estimate(small_model_config(num_layers=2), config)
    large = DenseSingleGPU().estimate(small_model_config(num_layers=6), config)

    assert large.total_memory_bytes > small.total_memory_bytes


def test_increasing_sequence_length_increases_activation_memory_monotonically():
    model_config = small_model_config()
    short = training_config(sequence_length=64)
    long = training_config(sequence_length=256)

    assert estimate_activation_memory_bytes(
        model_config, long
    ) > estimate_activation_memory_bytes(model_config, short)
