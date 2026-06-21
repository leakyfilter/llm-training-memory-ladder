# LLM Training Memory Ladder: Project Spec

## Goal

Build a local-first tool that explains how per-GPU training memory changes as we move through the LLM training parallelism ladder:

single GPU -> gradient accumulation -> DDP -> ZeRO-1 -> ZeRO-2 -> ZeRO-3/FSDP -> TP -> TP+SP.

The project should make the following idea concrete:

Different parallelism strategies attack different memory components.

- Gradient accumulation changes time/batch scheduling, not model-state memory.
- DDP replicates model states on every GPU.
- ZeRO-1 shards optimizer states.
- ZeRO-2 shards optimizer states and gradients.
- ZeRO-3/FSDP shards optimizer states, gradients and parameters.
- TP shards intra-layer weights/computation.
- SP shards some sequence-dimension activations, especially residual/norm/dropout regions.

## Non-goals for v0

- No real model training.
- No GPU requirement.
- No distributed PyTorch.
- No DeepSpeed.
- No FSDP implementation.
- No web UI.
- No tokenizer or dataset.

## v0 output

Given a Transformer model config, output a table comparing estimated per-GPU memory across strategies.

Example columns:

- strategy
- world_size
- tp_size
- dp_size
- parameter_memory_gb
- gradient_memory_gb
- optimizer_memory_gb
- activation_memory_gb
- total_memory_gb
- key_assumption

## v0 model config

Support:

- num_layers
- hidden_size
- sequence_length
- microbatch_size
- vocab_size
- mlp_expansion_ratio
- num_attention_heads
- dtype_bytes
- grad_bytes
- optimizer_state_bytes_per_param
- activation_multiplier_per_layer

## Approximate Transformer parameter model

For a decoder-only Transformer block:

Attention parameters per layer:

    q_proj: H * H
    k_proj: H * H
    v_proj: H * H
    o_proj: H * H

Total attention parameters:

    4 * H^2

SwiGLU MLP parameters per layer:

    gate_proj: H * I
    up_proj: H * I
    down_proj: I * H

Total MLP parameters:

    3 * H * I

where:

    I = mlp_expansion_ratio * H

Norm parameters are small but should be included as an optional term.

Token embeddings:

    vocab_size * H

LM head can be tied or untied. Default to tied.

## Memory accounting

Dense single-GPU model-state memory:

    params + gradients + optimizer_states

DDP per-GPU model-state memory:

    params + gradients + optimizer_states

ZeRO-1 per-GPU model-state memory:

    params + gradients + optimizer_states / dp_size

ZeRO-2 per-GPU model-state memory:

    params + gradients / dp_size + optimizer_states / dp_size

ZeRO-3/FSDP approximate per-GPU model-state memory:

    params / dp_size + gradients / dp_size + optimizer_states / dp_size + gather_buffer

Activation memory should be modeled separately because ZeRO does not automatically solve activation memory.

## Deliverables

- CLI or script that prints comparison tables.
- Plot showing memory by strategy.
- Tests validating formula behavior.
- Documentation of assumptions.