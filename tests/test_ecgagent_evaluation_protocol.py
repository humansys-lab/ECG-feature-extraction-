from __future__ import annotations

import json
import stat

import pytest

from ecgagent.evaluation_protocol import (
    _load_or_create_blinding_key,
    evaluate_arms,
)


def test_primary_endpoint_is_record_level_not_patient_union(tmp_path):
    truth = tmp_path / "truth.jsonl"
    truth.write_text(
        "\n".join(
            [
                json.dumps(
                    {"record_id": "r1", "patient_id": "p1", "labels": ["A"]}
                ),
                json.dumps(
                    {"record_id": "r2", "patient_id": "p1", "labels": ["B"]}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    arms = {}
    for role in ("rules", "single_turn", "agent"):
        path = tmp_path / f"{role}.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"record_id": "r1", "labels": ["B"]}),
                    json.dumps({"record_id": "r2", "labels": ["A"]}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        arms[role] = path

    report = evaluate_arms(
        truth,
        arms,
        blinding_key="test-key",
    )

    assert report["evaluation_unit"] == "record_id"
    assert report["patient_level"] is False
    for metrics in report["arms"].values():
        assert metrics["micro_f1"] == 0.0
        assert metrics["exact_match_rate"] == 0.0
        assert metrics["patient_aggregate_secondary"]["micro_f1"] == 1.0


def test_blinding_key_is_created_private_and_insecure_existing_key_fails(tmp_path):
    key_path = tmp_path / "private" / "blinding.key"

    created = _load_or_create_blinding_key(key_path, unblind=False)

    assert len(created) == 64
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert _load_or_create_blinding_key(key_path, unblind=True) == created

    key_path.chmod(0o644)
    with pytest.raises(ValueError, match="group/other"):
        _load_or_create_blinding_key(key_path, unblind=True)
