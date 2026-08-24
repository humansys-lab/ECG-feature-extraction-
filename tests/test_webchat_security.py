from __future__ import annotations

import asyncio
import io
import json
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from PIL import Image
from pydantic import ValidationError

import webchat.server as server
import app as legacy_app


SESSION_A = {"X-Chat-Session": "a" * 32}
SESSION_B = {"X-Chat-Session": "b" * 32}


def run_api(scenario) -> None:
    async def exercise() -> None:
        transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 43210))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            await scenario(client)

    asyncio.run(exercise())


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    server._clear_registered_attachments()
    server._RATE_BUCKETS.clear()
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "CHAT_AUTH_TOKEN", "")
    monkeypatch.setattr(server, "MAX_REQUESTS_PER_MINUTE", 120)
    monkeypatch.setattr(server, "_ACTIVE_CHATS", 0)
    monkeypatch.setattr(server, "_ACTIVE_UPLOADS", 0)
    yield
    server._clear_registered_attachments()
    server._RATE_BUCKETS.clear()


def test_chat_request_rejects_unbounded_and_unknown_inputs() -> None:
    with pytest.raises(ValidationError):
        server.ChatRequest(messages=[{"role": "user", "content": "ok"}], max_tokens=10**12)
    with pytest.raises(ValidationError):
        server.ChatRequest(messages=[{"role": "user", "content": "ok"}], temperature=2.1)
    with pytest.raises(ValidationError):
        server.ChatRequest(messages=[{"role": "system", "content": "ok"}])
    with pytest.raises(ValidationError):
        server.ChatRequest(messages=[{"role": "user", "content": "ok", "unknown": True}])
    with pytest.raises(ValidationError):
        server.ChatRequest(
            messages=[{"role": "user", "content": "x" * (server.MAX_MESSAGE_CHARS + 1)}]
        )


def test_api_authentication_and_browser_security_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "CHAT_AUTH_TOKEN", "correct-token-123456")
    async def scenario(client: httpx.AsyncClient) -> None:
        assert (await client.get("/api/config", headers=SESSION_A)).status_code == 401
        assert (
            (await client.get(
                "/api/config",
                headers={**SESSION_A, "X-Chat-Token": "incorrect-token"},
            )).status_code
            == 401
        )
        response = await client.get(
            "/api/config",
            headers={**SESSION_A, "X-Chat-Token": "correct-token-123456"},
        )
        assert response.status_code == 200

        page = await client.get("/")
        assert page.headers["x-content-type-options"] == "nosniff"
        assert "object-src 'none'" in page.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]

    run_api(scenario)


def test_host_and_origin_checks_block_dns_rebinding_and_cross_site_requests() -> None:
    async def exercise() -> None:
        transport = httpx.ASGITransport(
            app=server.app,
            client=("127.0.0.1", 43210),
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://evil.example",
        ) as rebound:
            response = await rebound.get("/api/config", headers=SESSION_A)
            assert response.status_code == 400
            assert "Host" in response.json()["detail"]

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:7861",
        ) as loopback:
            cross_site = await loopback.get(
                "/api/config",
                headers={**SESSION_A, "Origin": "https://evil.example"},
            )
            assert cross_site.status_code == 403
            assert "Origin" in cross_site.json()["detail"]

            wrong_port = await loopback.get(
                "/api/config",
                headers={**SESSION_A, "Origin": "http://127.0.0.1:9999"},
            )
            assert wrong_port.status_code == 403

            same_origin = await loopback.get(
                "/api/config",
                headers={**SESSION_A, "Origin": "http://127.0.0.1:7861"},
            )
            assert same_origin.status_code == 200

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as test_client:
            response = await test_client.get(
                "/api/config",
                headers={**SESSION_A, "Origin": "http://testserver"},
            )
            assert response.status_code == 200

    asyncio.run(exercise())


def test_remote_api_is_denied_when_no_token_is_configured() -> None:
    async def exercise() -> None:
        transport = httpx.ASGITransport(app=server.app, client=("203.0.113.9", 43210))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/config", headers=SESSION_A)
            assert response.status_code == 403
            assert response.headers["x-content-type-options"] == "nosniff"

    asyncio.run(exercise())


def test_model_upstream_requires_explicit_egress_and_never_inherits_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="CHAT_ALLOW_EXTERNAL_EGRESS"):
        server._enforce_upstream_privacy(
            "https://models.example/v1",
            allowed=False,
        )
    external = server._enforce_upstream_privacy(
        "https://models.example/v1",
        allowed=True,
    )
    assert external.external is True
    local = server._enforce_upstream_privacy(
        "http://127.0.0.1:8000/v1",
        allowed=False,
    )
    assert local.external is False
    with pytest.raises(RuntimeError, match="无凭据"):
        server._enforce_upstream_privacy(
            "https://user:secret@models.example/v1",
            allowed=True,
        )

    captured = {}
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    server._client()
    assert captured["trust_env"] is False


def test_attachment_filename_is_not_model_visible() -> None:
    identifier = "d" * 32
    server.ATTACHMENTS[identifier] = server.StoredAttachment(
        id=identifier,
        name="patient_John_Doe_12345.txt",
        kind="text",
        text="objective measurement",
        size_bytes=21,
        session_id="a" * 32,
    )

    parts = server._message_parts(
        server.ChatMessage(
            role="user",
            content="review",
            attachments=[identifier],
        ),
        "a" * 32,
    )

    rendered = json.dumps(parts, ensure_ascii=False)
    assert "objective measurement" in rendered
    assert "John_Doe" not in rendered


def test_text_attachment_is_private_and_delete_removes_disk_file() -> None:
    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/upload",
            headers=SESSION_A,
            files={"file": ("patient.txt", b"measured ECG data", "text/plain")},
            data={"strip_dx": "false"},
        )
        assert response.status_code == 200, response.text
        attachment_id = response.json()["id"]
        stored = server.ATTACHMENTS[attachment_id]
        assert stored.file_path is not None and stored.file_path.exists()
        assert stat.S_IMODE(stored.file_path.stat().st_mode) == 0o600

        assert (await client.get(f"/api/attachments/{attachment_id}", headers=SESSION_B)).status_code == 404
        deleted = await client.delete(f"/api/attachments/{attachment_id}", headers=SESSION_A)
        assert deleted.status_code == 200
        assert not stored.file_path.exists()
        assert not stored.file_path.parent.exists()

    run_api(scenario)


def test_upload_limit_is_enforced_during_stream_and_partial_file_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 4)
    async def scenario(client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/upload",
            headers=SESSION_A,
            files={"file": ("too-big.txt", b"12345", "text/plain")},
        )
        assert response.status_code == 413
        assert not list(server.UPLOAD_DIR.rglob("upload.txt"))
        assert not server.ATTACHMENTS

    run_api(scenario)


def test_image_content_is_verified_and_parse_failure_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "SUPPORTS_IMAGES", True)
    encoded = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(encoded, format="PNG")

    async def scenario(client: httpx.AsyncClient) -> None:
        mismatch = await client.post(
            "/api/upload",
            headers=SESSION_A,
            files={"file": ("fake.jpg", encoded.getvalue(), "image/jpeg")},
        )
        assert mismatch.status_code == 415
        assert not list(server.UPLOAD_DIR.rglob("upload.jpg"))

        corrupt = await client.post(
            "/api/upload",
            headers=SESSION_A,
            files={"file": ("broken.png", b"not a png", "image/png")},
        )
        assert corrupt.status_code == 415
        assert not list(server.UPLOAD_DIR.rglob("upload.png"))

    run_api(scenario)


def test_attachment_ttl_removes_registry_and_file(tmp_path: Path) -> None:
    directory = tmp_path / ("c" * 32)
    directory.mkdir()
    path = directory / "upload.txt"
    path.write_text("private", encoding="utf-8")
    stored = server.StoredAttachment(
        id="c" * 32,
        name="private.txt",
        kind="text",
        file_path=path,
        size_bytes=7,
        session_id="a" * 32,
        created_at=time.time() - server.ATTACHMENT_TTL_SECONDS - 1,
        last_accessed=time.time() - server.ATTACHMENT_TTL_SECONDS - 1,
    )
    server.ATTACHMENTS[stored.id] = stored
    with server._ATTACHMENT_LOCK:
        server._purge_expired_locked()
    assert stored.id not in server.ATTACHMENTS
    assert not path.exists()


def test_rate_and_generation_concurrency_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "MAX_REQUESTS_PER_MINUTE", 1)
    assert server._rate_allowed("client", now=10.0)
    assert not server._rate_allowed("client", now=11.0)
    assert server._rate_allowed("client", now=71.0)

    server._RATE_BUCKETS.clear()
    monkeypatch.setattr(server, "MAX_RATE_BUCKETS", 2)
    assert server._rate_allowed("one", now=100.0)
    assert server._rate_allowed("two", now=101.0)
    assert server._rate_allowed("three", now=102.0)
    assert len(server._RATE_BUCKETS) == 2
    assert "one" not in server._RATE_BUCKETS

    monkeypatch.setattr(server, "MAX_CONCURRENT_CHATS", 1)
    assert server._reserve_chat_slot()
    assert not server._reserve_chat_slot()
    server._release_chat_slot()

    monkeypatch.setattr(server, "MAX_CONCURRENT_UPLOADS", 1)
    assert server._reserve_upload_slot()
    assert not server._reserve_upload_slot()
    server._release_upload_slot()
    assert server._reserve_chat_slot()
    server._release_chat_slot()


def test_frontend_does_not_persist_medical_history_and_filters_link_protocols() -> None:
    app_source = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    markdown_bytes = (server.STATIC_DIR / "markdown.js").read_bytes()
    markdown_source = markdown_bytes.decode("utf-8")

    assert "sessionStorage.setItem(storageKey(STORAGE_KEY)" in app_source
    assert "localStorage.setItem(storageKey(STORAGE_KEY)" not in app_source
    assert "clearMedicalBrowserData" in app_source
    assert '["http:", "https:", "mailto:"]' in markdown_source
    assert 'class="unsafe-link"' in markdown_source
    assert b"\x00" not in markdown_bytes


def test_quality_stop_upload_is_never_sent_to_webchat_model(tmp_path: Path) -> None:
    payload = {
        "global_features": {},
        "interpretation": {},
        "metadata": {
            "diagnostic_gate": {
                "state": "stop",
                "stop_reasons": ["insufficient_beats"],
                "partial_reasons": [],
                "allowed_domains": [],
                "suppressed_domains": ["all"],
            }
        },
    }
    async def scenario(client: httpx.AsyncClient) -> None:
        uploaded = await client.post(
            "/api/upload",
            headers=SESSION_A,
            files={
                "file": (
                    "stopped_features.json",
                    json.dumps(payload).encode("utf-8"),
                    "application/json",
                )
            },
        )
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()["diagnostic_gate_state"] == "stop"

        response = await client.post(
            "/api/chat",
            headers=SESSION_A,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "请诊断",
                        "attachments": [uploaded.json()["id"]],
                    }
                ]
            },
        )
        assert response.status_code == 422
        assert "STOP" in response.json()["detail"]

    run_api(scenario)


def test_partial_gate_is_blocked_by_single_shot_web_surfaces(tmp_path: Path) -> None:
    payload = {
        "global_features": {},
        "interpretation": {},
        "metadata": {
            "diagnostic_gate": {
                "state": "partial",
                "stop_reasons": [],
                "partial_reasons": ["voltage_domain_suppressed"],
                "allowed_domains": ["rhythm"],
                "suppressed_domains": ["voltage"],
            }
        },
    }

    async def scenario(client: httpx.AsyncClient) -> None:
        uploaded = await client.post(
            "/api/upload",
            headers=SESSION_A,
            files={
                "file": (
                    "partial_features.json",
                    json.dumps(payload).encode("utf-8"),
                    "application/json",
                )
            },
        )
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()["diagnostic_gate_state"] == "partial"
        response = await client.post(
            "/api/chat",
            headers=SESSION_A,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "请诊断",
                        "attachments": [uploaded.json()["id"]],
                    }
                ]
            },
        )
        assert response.status_code == 422
        assert "PARTIAL" in response.json()["detail"]
        assert "voltage_domain_suppressed" in response.json()["detail"]

    run_api(scenario)

    path = tmp_path / "partial_features.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[str] = []
    result = legacy_app.guarded_diagnosis(
        SimpleNamespace(name=str(path)),
        None,
        "",
        "en",
        lambda prompt: calls.append(prompt) or "must not run",
    )
    assert calls == []
    assert result[1] == ""
    assert "PARTIAL" in result[2]


def test_renaming_feature_json_cannot_bypass_webchat_gate() -> None:
    payload = {
        "global_features": {},
        "interpretation": {},
        "metadata": {
            "diagnostic_gate": {
                "state": "stop",
                "stop_reasons": ["insufficient_beats"],
                "partial_reasons": [],
                "allowed_domains": [],
                "suppressed_domains": ["all"],
            }
        },
    }

    async def scenario(client: httpx.AsyncClient) -> None:
        uploaded = await client.post(
            "/api/upload",
            headers=SESSION_A,
            files={
                "file": (
                    "renamed_record.txt",
                    json.dumps(payload).encode("utf-8"),
                    "text/plain",
                )
            },
        )
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()["diagnostic_gate_state"] == "stop"

        response = await client.post(
            "/api/chat",
            headers=SESSION_A,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "请诊断",
                        "attachments": [uploaded.json()["id"]],
                    }
                ]
            },
        )
        assert response.status_code == 422
        assert "STOP" in response.json()["detail"]

    run_api(scenario)


def test_single_shot_gate_state_is_fail_closed_and_pass_is_allowed(tmp_path: Path) -> None:
    unknown_id = "e" * 32
    server.ATTACHMENTS[unknown_id] = server.StoredAttachment(
        id=unknown_id,
        name="features.json",
        kind="ecg_features",
        text="objective measurement",
        size_bytes=21,
        session_id="a" * 32,
        diagnostic_gate_state="future_state",
    )
    with pytest.raises(server.HTTPException) as exc_info:
        server._message_parts(
            server.ChatMessage(role="user", content="review", attachments=[unknown_id]),
            "a" * 32,
        )
    assert exc_info.value.status_code == 422

    pass_id = "f" * 32
    server.ATTACHMENTS[pass_id] = server.StoredAttachment(
        id=pass_id,
        name="features.json",
        kind="ecg_features",
        text="objective measurement",
        size_bytes=21,
        session_id="a" * 32,
        diagnostic_gate_state="pass",
    )
    assert server._message_parts(
        server.ChatMessage(role="user", content="review", attachments=[pass_id]),
        "a" * 32,
    )

    path = tmp_path / "passing_features.json"
    path.write_text(
        json.dumps(
            {
                "global_features": {},
                "interpretation": {},
                "metadata": {
                    "diagnostic_gate": {
                        "state": "pass",
                        "stop_reasons": [],
                        "partial_reasons": [],
                        "allowed_domains": ["all"],
                        "suppressed_domains": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []
    result = legacy_app.guarded_diagnosis(
        SimpleNamespace(name=str(path)),
        None,
        "",
        "en",
        lambda prompt: calls.append(prompt) or "model result",
    )
    assert len(calls) == 1
    assert result[2] == "model result"


def test_gradio_guard_abstains_without_calling_generator(tmp_path: Path) -> None:
    path = tmp_path / "stopped_features.json"
    path.write_text(
        json.dumps(
            {
                "global_features": {},
                "interpretation": {},
                "metadata": {
                    "diagnostic_gate": {
                        "state": "stop",
                        "stop_reasons": ["channel_desynchronized"],
                        "partial_reasons": [],
                        "allowed_domains": [],
                        "suppressed_domains": ["all"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    result = legacy_app.guarded_diagnosis(
        SimpleNamespace(name=str(path)),
        None,
        "",
        "en",
        lambda prompt: calls.append(prompt) or "must not run",
    )

    assert calls == []
    assert result[1] == ""
    assert "ABSTAIN" in result[2]
    assert "channel_desynchronized" in result[2]


def test_missing_webchat_gate_is_stopped_before_upstream_model() -> None:
    payload = {
        "global_features": {},
        "interpretation": {},
        "metadata": {},
    }

    async def scenario(client: httpx.AsyncClient) -> None:
        uploaded = await client.post(
            "/api/upload",
            headers=SESSION_A,
            files={
                "file": (
                    "missing_gate_features.json",
                    json.dumps(payload).encode("utf-8"),
                    "application/json",
                )
            },
        )
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()["diagnostic_gate_state"] == "stop"
        response = await client.post(
            "/api/chat",
            headers=SESSION_A,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "请诊断",
                        "attachments": [uploaded.json()["id"]],
                    }
                ]
            },
        )
        assert response.status_code == 422
        assert "STOP" in response.json()["detail"]

    run_api(scenario)


def test_malformed_gradio_gate_abstains_without_model_call(tmp_path: Path) -> None:
    path = tmp_path / "malformed_features.json"
    path.write_text(
        json.dumps(
            {
                "global_features": {},
                "interpretation": {},
                "metadata": {"diagnostic_gate": "pass"},
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    result = legacy_app.guarded_diagnosis(
        SimpleNamespace(name=str(path)),
        None,
        "",
        "en",
        lambda prompt: calls.append(prompt) or "must not run",
    )

    assert calls == []
    assert result[1] == ""
    assert "ABSTAIN" in result[2]
    assert "malformed_diagnostic_gate" in result[2]
