import json

from hermes_cli.model_switch import (
    _model_router_catalog_accepts,
    list_picker_providers,
)


def _write_catalog(path):
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "openai-codex": {
                        "gpt-5.3-codex": {},
                        "gpt-5.4": {},
                    },
                    "github-copilot": {
                        "grok-code-fast-1": {},
                    },
                    "ollama": {
                        "gemma4:26b": {},
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_model_router_catalog_filters_picker_rows(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    monkeypatch.setenv("HERMES_MODEL_ROUTER_CATALOG", str(catalog_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    providers = list_picker_providers(
        current_provider="openai-codex",
        user_providers={
            "openai-codex": {
                "name": "OpenAI Codex",
                "api": "https://example.invalid/v1",
                "default_model": "gpt-5.5-pro",
                "models": ["gpt-5.5-pro", "gpt-5.3-codex", "gpt-5.4"],
            },
            "copilot": {
                "name": "GitHub Copilot",
                "api": "https://example.invalid/v1",
                "default_model": "claude-sonnet-4.5",
                "models": ["claude-sonnet-4.5", "grok-code-fast-1"],
            },
        },
        custom_providers=[],
        max_models=50,
    )

    by_catalog = {row.get("catalog_provider"): row for row in providers if row.get("source") == "model-router-catalog"}

    assert by_catalog["openai-codex"]["models"] == ["gpt-5.3-codex", "gpt-5.4"]
    assert by_catalog["openai-codex"]["total_models"] == 2
    assert by_catalog["github-copilot"]["models"] == ["grok-code-fast-1"]
    assert all("gpt-5.5-pro" not in row.get("models", []) for row in providers)


def test_model_router_catalog_keeps_user_defined_unmatched_provider(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    monkeypatch.setenv("HERMES_MODEL_ROUTER_CATALOG", str(catalog_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    providers = list_picker_providers(
        current_provider="private-provider",
        user_providers={
            "private-provider": {
                "name": "Private Provider",
                "api": "https://example.invalid/v1",
                "default_model": "internal-model",
                "models": ["internal-model"],
            }
        },
        custom_providers=[],
        max_models=50,
    )

    private_row = next(row for row in providers if row.get("slug") == "private-provider")
    assert private_row["models"] == ["internal-model"]
    assert private_row["is_user_defined"] is True


def test_model_router_catalog_accepts_prefixed_and_provider_local_ids(monkeypatch, tmp_path):
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    monkeypatch.setenv("HERMES_MODEL_ROUTER_CATALOG", str(catalog_path))

    assert _model_router_catalog_accepts("openai-codex", "gpt-5.3-codex") is True
    assert _model_router_catalog_accepts("openai-codex", "openai-codex/gpt-5.3-codex") is True
    assert _model_router_catalog_accepts("copilot", "grok-code-fast-1") is True
    assert _model_router_catalog_accepts("openai-codex", "gpt-5.5-pro") is False
