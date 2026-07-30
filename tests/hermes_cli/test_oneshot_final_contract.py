"""Regression coverage for one-shot final-response evidence."""


def test_oneshot_fails_closed_when_failed_with_error_text(monkeypatch, capsys):
    from hermes_cli.oneshot import run_oneshot

    monkeypatch.setattr(
        "hermes_cli.oneshot._run_agent",
        lambda *_args, **_kwargs: (
            "API call failed after 3 retries: HTTP 404: model not found",
            {"failed": True, "partial": False},
        ),
    )

    assert run_oneshot("hi") == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "did not complete" in captured.err


def test_oneshot_fails_closed_on_partial_with_commentary(monkeypatch, capsys):
    from hermes_cli.oneshot import run_oneshot

    monkeypatch.setattr(
        "hermes_cli.oneshot._run_agent",
        lambda *_args, **_kwargs: (
            "I am still checking the sources.",
            {"failed": False, "partial": True, "completed": False},
        ),
    )

    assert run_oneshot("hi") == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "did not complete" in captured.err
