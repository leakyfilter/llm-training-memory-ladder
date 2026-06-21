# AGENTS.md

## Project

This repository implements "LLM Training Memory Ladder", a local-first educational systems project for modeling memory tradeoffs in LLM training parallelism.

The goal is not to train a useful model. The goal is to make memory and communication tradeoffs visible and testable.

## Core concepts

The project should progressively model:

1. Single-GPU training memory
2. Gradient accumulation
3. Data Parallelism / DDP
4. ZeRO-1
5. ZeRO-2
6. ZeRO-3 / FSDP-style sharding
7. Tensor Parallelism
8. Sequence Parallelism
9. Activation checkpointing
10. Later: Context Parallelism and Pipeline Parallelism

## Development rules

- Prefer small, testable changes.
- Do not implement distributed training unless explicitly asked.
- Do not add PyTorch unless explicitly asked.
- Do not add a web app unless explicitly asked.
- Do not silently invent formulas.
- Every formula must have comments explaining the assumption.
- Every memory quantity must specify units.
- Every strategy must document which tensors are replicated and which are sharded.
- Favor simple dataclasses over complex abstractions.
- Keep plots publication-quality but minimal.
- Use seaborn for static plots and plotly for interactive plots.
- Add or update tests for every formula change.

## Commands

Use these commands to validate changes:

```bash
uv sync
uv run pytest
```

If plotting code is changed, also run:
```bash
uv run python -m memory_ladder.examples.basic_memory_sweep
```

## Style

- Python package under src/memory_ladder.
- Tests under tests.
- Use clear names: param_bytes, grad_bytes, optimizer_bytes, activation_bytes.
- Avoid abbreviations unless they are standard: DP, DDP, TP, SP, FSDP.
- Use explicit units in function names or return schemas when possible.