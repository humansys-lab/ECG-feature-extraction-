"""Guard behaviour that must degrade a run instead of discarding it.

Every case here maps to a record lost in the ptbxl_09000 qwen_v31 batch, where
83% of runs ended at a guard rather than at a clinical conclusion.
"""
from __future__ import annotations

from ecgagent.agent import diagnostic_prompts
from ecgagent.agent.diagnostic import ECGDiagnosticAgent
from ecgagent.agent.diagnostic_ledger import DiagnosticLedger
from ecgagent.agent.diagnosis_catalog import allowed_diagnosis_codes
from ecgagent.agent.protocol import DIAGNOSTIC_DOMAINS, DIAGNOSTIC_TOOLS
from ecgagent.backends.mock import ScriptedBackend
from ecgagent.evidence.store import EvidenceStore
from tests.test_ecgagent_tools import _payload


HEART_RATE = "/global_features/heart_rate_bpm"
QRS = "/global_features/qrs_ms"
PR = "/global_features/pr_ms"


def _ledger() -> DiagnosticLedger:
    return DiagnosticLedger(
        store=EvidenceStore.from_dict(_payload(), record_id="LR001"),
        diagnosis_codes=allowed_diagnosis_codes(),
        domains=DIAGNOSTIC_DOMAINS,
        tools=DIAGNOSTIC_TOOLS,
    )


def _domains(pointer: str = HEART_RATE) -> dict:
    return {
        domain: {
            "status": "assessed",
            "finding": "normal",
            "citations": [f"ev:{pointer}"],
            "limitations": [],
        }
        for domain in DIAGNOSTIC_DOMAINS
    }


def _hypothesis(hypothesis_id: str, code: str, pointer: str) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "diagnosis_code": code,
        "phenotype": f"{code} candidate",
        "kind": "diagnosis",
        "domains": ["rhythm_rate"],
        "status": "raised",
        "supporting_observations": [{"citations": [f"ev:{pointer}"]}],
        "counterevidence": [],
        "required_evidence": ["confirm against a second view"],
        "next_tools": ["get_measurement"],
        "alternative_explanations": ["measurement artefact"],
    }


def _survey(ledger: DiagnosticLedger, *, hypotheses: list[dict]) -> None:
    ledger.apply_phase_response(
        "survey",
        {
            "base_version": 0,
            "domains": _domains(),
            "create_hypotheses": hypotheses,
            "evidence_updates": [],
            "domain_updates": [],
            "transitions": [],
            "plan_updates": [],
            "refuting_tests": [],
            "targeted_checks": [],
            "detector_conflicts": [],
            "unresolved": [],
        },
        authorized_citations=[HEART_RATE, QRS, PR],
        new_phase_citations=[HEART_RATE],
    )


def test_ledger_accepts_the_evidence_ids_it_published_to_the_model():
    """The snapshot keys evidence by content id, so that id must cite."""

    ledger = _ledger()
    _survey(ledger, hypotheses=[_hypothesis("h_rate", "bradycardia", HEART_RATE)])

    published = ledger.phase_state()["evidence"]
    evidence_id = next(
        key
        for key, row in published.items()
        if row.get("pointer") == HEART_RATE
    )
    assert published[evidence_id]["quality_status"] in {"reliable", "limited"}
    assert "missing_reason" in published[evidence_id]
    assert "scope" in published[evidence_id]
    assert "evidence_contract_version" in published[evidence_id]
    assert evidence_id.startswith("E")

    resolved = ledger.resolve_citation(evidence_id)
    assert resolved is not None and resolved.pointer == HEART_RATE

    problems = ledger.validate_phase_response(
        "hypothesize",
        {
            "base_version": ledger.version,
            "create_hypotheses": [],
            "evidence_updates": [],
            "domain_updates": [],
            "transitions": [
                {
                    "hypothesis_id": "h_rate",
                    "from_status": "raised",
                    "to_status": "provisional",
                    "citations": [evidence_id],
                }
            ],
            "plan_updates": [],
            "refuting_tests": [],
            "targeted_checks": [],
            "detector_conflicts": [],
            "unresolved": [],
        },
        authorized_citations=[HEART_RATE],
        new_phase_citations=[],
    )
    assert problems == []


def test_unresolvable_citation_is_still_reported():
    ledger = _ledger()
    _survey(ledger, hypotheses=[_hypothesis("h_rate", "bradycardia", HEART_RATE)])

    problems = ledger.validate_phase_response(
        "hypothesize",
        {
            "base_version": ledger.version,
            "create_hypotheses": [],
            "evidence_updates": [],
            "domain_updates": [],
            "transitions": [
                {
                    "hypothesis_id": "h_rate",
                    "from_status": "raised",
                    "to_status": "provisional",
                    "citations": ["E0123456789abcdef"],
                }
            ],
            "plan_updates": [],
            "refuting_tests": [],
            "targeted_checks": [],
            "detector_conflicts": [],
            "unresolved": [],
        },
        authorized_citations=[HEART_RATE],
        new_phase_citations=[],
    )
    assert any("unresolvable citation" in problem for problem in problems)


def test_new_hypothesis_may_cite_evidence_read_in_an_earlier_phase():
    """QRS duration is the natural counterevidence to a fascicular block."""

    ledger = _ledger()
    _survey(ledger, hypotheses=[_hypothesis("h_rate", "bradycardia", HEART_RATE)])

    late_candidate = _hypothesis("h_lafb", "lafb_pattern", PR)
    late_candidate["domains"] = ["axis"]
    late_candidate["counterevidence"] = [{"citations": [f"ev:{QRS}"]}]

    patch = {
        "base_version": ledger.version,
        "create_hypotheses": [late_candidate],
        "evidence_updates": [],
        "domain_updates": [],
        "transitions": [],
        "plan_updates": [],
        "refuting_tests": [],
        "targeted_checks": [],
        "detector_conflicts": [],
        "unresolved": [],
    }
    # Only PR was read in this phase; QRS carries over from Survey.
    problems = ledger.validate_phase_response(
        "investigate",
        patch,
        authorized_citations=[HEART_RATE, QRS, PR],
        new_phase_citations=[PR],
    )
    assert problems == []

    ledger.apply_phase_response(
        "investigate",
        patch,
        authorized_citations=[HEART_RATE, QRS, PR],
        new_phase_citations=[PR],
    )
    assert ledger.hypotheses["h_lafb"]["status"] == "raised"


def test_unchallenged_support_is_downgraded_rather_than_aborting_the_run():
    ledger = _ledger()
    _survey(ledger, hypotheses=[_hypothesis("h_rate", "bradycardia", HEART_RATE)])
    ledger.apply_phase_response(
        "hypothesize",
        {
            "base_version": ledger.version,
            "create_hypotheses": [],
            "evidence_updates": [],
            "domain_updates": [],
            "transitions": [
                {
                    "hypothesis_id": "h_rate",
                    "from_status": "raised",
                    "to_status": "provisional",
                    "citations": [f"ev:{HEART_RATE}"],
                }
            ],
            "plan_updates": [],
            "refuting_tests": [],
            "targeted_checks": [],
            "detector_conflicts": [],
            "unresolved": [],
        },
        authorized_citations=[HEART_RATE, QRS, PR],
        new_phase_citations=[],
    )

    # Challenge promotes to supported but files no refuting test at all.
    ledger.apply_phase_response(
        "challenge",
        {
            "base_version": ledger.version,
            "create_hypotheses": [],
            "evidence_updates": [],
            "domain_updates": [],
            "transitions": [
                {
                    "hypothesis_id": "h_rate",
                    "from_status": "provisional",
                    "to_status": "supported",
                    "citations": [f"ev:{QRS}"],
                }
            ],
            "plan_updates": [],
            "refuting_tests": [],
            "targeted_checks": [],
            "detector_conflicts": [],
            "unresolved": [],
        },
        authorized_citations=[HEART_RATE, QRS, PR],
        new_phase_citations=[QRS],
    )

    assert ledger.hypotheses["h_rate"]["status"] == "unresolved"
    assert ledger.frozen
    assert any(
        event["op"] == "automatic_safety_downgrade"
        for event in ledger.history
    )
    assert "bradycardia" not in ledger.placement_contract().positive_codes


def test_unchallenged_support_is_classified_as_deterministically_remediated():
    problems = diagnostic_prompts.validate_phase_state(
        "challenge",
        {
            "base_version": 3,
            "hypotheses": [],
            "create_hypotheses": [],
            "transitions": [],
            "refuting_tests": [],
        },
        ledger_state={
            "version": 3,
            "hypotheses": [
                {
                    "id": "h_axis_review",
                    "diagnosis_code": "left_axis_deviation",
                    "status": "supported",
                }
            ],
        },
    )

    assert any("no discriminative Challenge falsification test" in row for row in problems)
    assert all(
        diagnostic_prompts.is_remediated_phase_problem(row) for row in problems
    )


def test_an_unrepairable_defect_stays_fatal():
    assert not diagnostic_prompts.is_remediated_phase_problem(
        "transitions[0] has unresolvable citation `E0123456789abcdef`"
    )
    assert not diagnostic_prompts.is_remediated_phase_problem(
        "create_hypotheses[0] has an unregistered diagnosis_code"
    )


def test_agent_partitions_guard_problems_into_fatal_and_remediated():
    """The classifier must actually be reachable from the diagnostic profile."""

    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="LR002"),
        backend=ScriptedBackend(script=[]),
    )

    remediated = (
        "supported hypothesis `h_axis_review` has no discriminative "
        "Challenge falsification test"
    )
    unrepairable = "transitions[0] has unresolvable citation `E0123456789abcdef`"

    assert agent._fatal_phase_problems([remediated]) == []
    assert agent._fatal_phase_problems([remediated, unrepairable]) == [unrepairable]


def test_a_profile_without_the_hook_keeps_every_problem_fatal():
    class _NoHook:
        pass

    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="LR003"),
        backend=ScriptedBackend(script=[]),
    )
    agent.prompt_module = _NoHook()

    problems = [
        "supported hypothesis `h` has no discriminative Challenge falsification test"
    ]
    assert agent._fatal_phase_problems(problems) == problems


def test_ledger_evidence_ids_are_declared_as_citation_tokens():
    """Otherwise the sanitizer reads a copied-out id as citing nothing."""

    agent = ECGDiagnosticAgent(
        workflow="legacy",
        store=EvidenceStore.from_dict(_payload(), record_id="LR004"),
        backend=ScriptedBackend(script=[]),
    )
    _survey(
        agent._diagnostic_ledger,
        hypotheses=[_hypothesis("h_rate", "bradycardia", HEART_RATE)],
    )

    aliases = agent._state_citation_aliases()
    assert HEART_RATE in aliases.values()
    assert all(key.startswith("E") for key in aliases)
    assert "bradycardia" in agent._active_state_codes()


def test_sanitizer_drops_a_second_hypothesis_for_an_already_carried_code():
    """`normal intervals` and `normal axis` both fall back to `sinus_rhythm`."""

    state = {
        "create_hypotheses": [
            {"hypothesis_id": "h_sinus", "diagnosis_code": "sinus_rhythm"},
            {"hypothesis_id": "h_intervals", "diagnosis_code": "sinus_rhythm"},
            {"hypothesis_id": "h_axis", "diagnosis_code": "sinus_rhythm"},
            {"hypothesis_id": "h_brady", "diagnosis_code": "bradycardia"},
        ],
        "transitions": [],
    }

    sanitized, notes = diagnostic_prompts.sanitize_phase_state("survey", state)

    kept = [row["hypothesis_id"] for row in sanitized["create_hypotheses"]]
    assert kept == ["h_sinus", "h_brady"]
    assert sum("already has an active hypothesis" in note for note in notes) == 2


def test_sanitizer_also_drops_a_code_the_ledger_already_carries():
    state = {
        "create_hypotheses": [
            {"hypothesis_id": "h_again", "diagnosis_code": "bradycardia"},
            {"hypothesis_id": "h_new", "diagnosis_code": "prolonged_qt"},
        ],
        "transitions": [],
    }

    sanitized, _ = diagnostic_prompts.sanitize_phase_state(
        "investigate",
        state,
        active_diagnosis_codes=("bradycardia",),
    )

    kept = [row["hypothesis_id"] for row in sanitized["create_hypotheses"]]
    assert kept == ["h_new"]


def test_survey_sanitizer_drops_only_candidates_without_intake_evidence():
    state = {
        "base_version": 9,
        "create_hypotheses": [
            {
                "hypothesis_id": "h_supported_by_intake",
                "diagnosis_code": "bradycardia",
                "status": "raised",
                "supporting_observations": [{"citations": [f"ev:{HEART_RATE}"]}],
                "counterevidence": [],
            },
            {
                "hypothesis_id": "h_without_evidence",
                "diagnosis_code": "tachycardia",
                "status": "raised",
                "supporting_observations": [],
                "counterevidence": [],
            },
        ],
        "transitions": [],
    }

    sanitized, notes = diagnostic_prompts.sanitize_phase_state(
        "survey",
        state,
        phase_citations=(f"ev:{HEART_RATE}",),
        expected_base_version=0,
    )

    assert sanitized["base_version"] == 0
    assert [
        row["hypothesis_id"] for row in sanitized["create_hypotheses"]
    ] == ["h_supported_by_intake"]
    assert any("program-owned" in note for note in notes)
    assert any("no citation" in note for note in notes)


def test_normalizer_stamps_the_category_the_catalog_defines():
    verdict = {
        "summary": "",
        "diagnoses": [
            {
                "code": "p_wave_abnormality",
                "category": "axis",
                "statement": "P 波形态异常。",
                "confidence": "MEDIUM",
                "urgency": "NONE",
                "evidence": [],
                "counterevidence": [],
                "reasoning": "",
            }
        ],
        "differential_diagnoses": [],
    }

    normalized, changed = diagnostic_prompts.normalize_verdict(verdict)

    assert normalized["diagnoses"][0]["category"] == "rhythm"
    assert "diagnoses[0].category" in changed
    assert not any(
        "conflicts with code" in problem
        for problem in diagnostic_prompts.validate_verdict(
            normalized,
            evidence_document=_payload(),
        )
    )
