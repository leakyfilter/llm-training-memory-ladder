# LLM Training Memory Ladder

Local-first educational tooling for estimating per-GPU training memory across
LLM parallelism strategies.

This v0 is intentionally pure Python. It does not train models, use PyTorch, or
implement distributed execution.

## Validate

```bash
uv sync
uv run pytest
uv run python -m memory_ladder.examples.basic_memory_sweep
```
