import main


def test_main_without_command_prints_help(capsys):
    result = main.main([])

    captured = capsys.readouterr()
    assert result == 0
    assert "usage:" in captured.out
    assert "debug" in captured.out
    assert "sanity" in captured.out
    assert "smoke" in captured.out


def test_debug_command_forwards_args(monkeypatch):
    calls = []

    def fake_run_module_main(module_name, forwarded_args):
        calls.append((module_name, forwarded_args))
        return 7

    monkeypatch.setattr(main, "_run_module_main", fake_run_module_main)

    result = main.main(["debug", "--target", "wrapper", "--duration", "1"])

    assert result == 7
    assert calls == [
        (
            "examples.debug_highway_env_episode",
            ["--target", "wrapper", "--duration", "1"],
        )
    ]


def test_sanity_command_forwards_args(monkeypatch):
    calls = []

    def fake_run_module_main(module_name, forwarded_args):
        calls.append((module_name, forwarded_args))
        return 3

    monkeypatch.setattr(main, "_run_module_main", fake_run_module_main)

    result = main.main(["sanity", "--policy", "random"])

    assert result == 3
    assert calls == [
        (
            "examples.random_highway_intersection",
            ["--policy", "random"],
        )
    ]


def test_smoke_command_runs_explicit_smoke_check(monkeypatch):
    calls = []

    def fake_run_smoke_check():
        calls.append("smoke")
        return 0

    monkeypatch.setattr(main, "run_smoke_check", fake_run_smoke_check)

    result = main.main(["smoke"])

    assert result == 0
    assert calls == ["smoke"]
