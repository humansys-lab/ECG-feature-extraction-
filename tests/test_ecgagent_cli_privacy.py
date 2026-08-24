from __future__ import annotations

import json

from ecgagent import cli
from ecgagent.cli import (
    CLI_PSEUDONYM_STRATEGY,
    _external_egress_error,
    _pseudonymized_agent_store,
)
from ecgagent.evidence.store import EvidenceStore
from ecgagent.privacy import (
    effective_qwen_base_url,
    endpoint_is_literal_loopback,
    resolve_backend_privacy,
)


def test_cloud_backend_requires_explicit_external_egress_authorization() -> None:
    error = _external_egress_error("anthropic", allowed=False)

    assert error is not None
    assert "--allow-external-egress" in error
    assert _external_egress_error("deepseek", allowed=False) is not None


def test_local_backends_never_require_external_egress_authorization() -> None:
    assert _external_egress_error("medgemma-local", allowed=False) is None
    assert _external_egress_error("qwen-local", allowed=False) is None
    assert _external_egress_error("anthropic", allowed=True) is None


def test_qwen_effective_endpoint_precedence_and_remote_gate(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_BASE_URL", "https://remote.example/v1")
    assert effective_qwen_base_url() == (
        "https://remote.example/v1",
        "environment",
    )
    assert effective_qwen_base_url("http://127.0.0.1:9000/v1") == (
        "http://127.0.0.1:9000/v1",
        "explicit",
    )
    assert _external_egress_error("qwen", allowed=False) is not None
    assert (
        _external_egress_error(
            "qwen-local",
            allowed=False,
            qwen_base_url="http://[::1]:8000/v1",
        )
        is None
    )


def test_only_literal_loopback_and_local_socket_endpoints_are_local() -> None:
    assert endpoint_is_literal_loopback("http://localhost:8000/v1")
    assert endpoint_is_literal_loopback("https://127.0.0.2/v1")
    assert endpoint_is_literal_loopback("unix:///tmp/qwen.sock")
    assert endpoint_is_literal_loopback("http+unix://%2Ftmp%2Fqwen.sock/v1")
    assert not endpoint_is_literal_loopback("http://localhost.example/v1")
    assert not endpoint_is_literal_loopback("http://qwen.internal/v1")
    assert not endpoint_is_literal_loopback("not-a-url")


def test_endpoint_audit_redacts_credentials_and_query(monkeypatch) -> None:
    endpoint = "https://alice:secret@example.com:9443/v1?api_key=also-secret"
    policy = resolve_backend_privacy("qwen-local", qwen_base_url=endpoint)
    serialized = json.dumps(policy.audit(allowed=True), sort_keys=True)

    assert policy.external is True
    assert "https://example.com:9443" in serialized
    assert "alice" not in serialized
    assert "secret" not in serialized
    assert "api_key" not in serialized

    monkeypatch.setenv("ANTHROPIC_BASE_URL", endpoint)
    anthropic = resolve_backend_privacy("anthropic")
    assert anthropic.effective_endpoint == endpoint
    assert anthropic.endpoint_source == "environment"
    assert "secret" not in json.dumps(anthropic.audit(allowed=True))


def test_cli_blocks_remote_qwen_from_environment_before_backend_start(
    tmp_path,
    monkeypatch,
) -> None:
    feature_path = tmp_path / "sample_features.json"
    feature_path.write_text(
        json.dumps(
            {
                "global_features": {},
                "interpretation": {},
                "clinical_interpretation": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QWEN_BASE_URL", "https://remote.example/v1")

    assert (
        cli.main(
            [
                "--features",
                str(feature_path),
                "--agent",
                "--backend",
                "qwen-local",
                "--quiet",
            ]
        )
        == 2
    )


def test_cli_agent_store_uses_ephemeral_pseudonym_without_persisting_key() -> None:
    real_id = "sensitive-record-123"
    source = EvidenceStore.from_dict(
        {
            "record_id": real_id,
            "metadata": {"record": real_id},
            "global_features": {},
            "interpretation": {},
        },
        record_id=real_id,
    )

    model_store, audit = _pseudonymized_agent_store(source)

    assert model_store.record_id.startswith("ecg_")
    assert model_store.record_id != real_id
    assert model_store.document["record_id"] == model_store.record_id
    assert model_store.document["metadata"]["record"] == model_store.record_id
    assert source.document["record_id"] == real_id
    assert audit["strategy"] == CLI_PSEUDONYM_STRATEGY
    assert real_id not in json.dumps(audit)
    assert set(audit) == {
        "model_visible_record_id",
        "strategy",
        "key_scope",
    }
