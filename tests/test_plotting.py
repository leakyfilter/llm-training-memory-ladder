from memory_ladder.examples.plot_memory_ladder import main


def test_plot_memory_ladder_script_creates_output_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    main()

    output_path = tmp_path / "outputs" / "memory_by_strategy.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
