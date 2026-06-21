from memory_ladder import DenseSingleGPU, TrainingConfig, TransformerConfig, ZeRO3
from memory_ladder.sweeps import sweep_sequence_lengths, sweep_shapes


def test_sweep_shapes_generates_rows_for_each_shape_and_strategy():
    rows = sweep_shapes(
        base_model_config=TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=8,
            vocab_size=256,
        ),
        base_training_config=TrainingConfig(
            sequence_length=32,
            microbatch_size=1,
            dp_size=4,
        ),
        hidden_sizes=[64, 128],
        num_layers_values=[2],
        sequence_lengths=[32, 64],
        microbatch_sizes=[1, 2],
        strategies=[DenseSingleGPU(), ZeRO3()],
    )

    assert len(rows) == 2 * 1 * 2 * 2 * 2
    assert {row["hidden_size"] for row in rows} == {64, 128}
    assert {row["sequence_length"] for row in rows} == {32, 64}
    assert {row["microbatch_size"] for row in rows} == {1, 2}


def test_sweep_marks_budget_oom_and_winner_per_shape():
    rows = sweep_sequence_lengths(
        base_model_config=TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=8,
            vocab_size=256,
        ),
        base_training_config=TrainingConfig(
            sequence_length=32,
            microbatch_size=1,
            dp_size=4,
        ),
        sequence_lengths=[32],
        gpu_memory_budget_gb=0.0001,
        strategies=[DenseSingleGPU(), ZeRO3()],
    )

    assert all(row["oom"] for row in rows)
    winners = [row for row in rows if row["is_winner"]]
    assert len(winners) == 1
    assert winners[0]["winning_strategy"] == winners[0]["strategy"]
    assert winners[0]["total_memory_bytes"] == min(
        row["total_memory_bytes"] for row in rows
    )


def test_sequence_length_sweep_increases_activation_memory_for_each_strategy():
    rows = sweep_sequence_lengths(
        base_model_config=TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=8,
            vocab_size=256,
        ),
        base_training_config=TrainingConfig(
            sequence_length=32,
            microbatch_size=1,
            dp_size=4,
        ),
        sequence_lengths=[32, 64],
        strategies=[DenseSingleGPU(), ZeRO3()],
    )

    by_strategy = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)

    for strategy_rows in by_strategy.values():
        strategy_rows.sort(key=lambda row: row["sequence_length"])
        assert (
            strategy_rows[1]["activation_memory_bytes"]
            > strategy_rows[0]["activation_memory_bytes"]
        )
