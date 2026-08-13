"""Diagnosis-first LLM agent over the deterministic ecgfeat measurement layer.

`ecgfeat` is the source of measured values, provenance and reliability flags.
The language model independently interprets those observations and produces
diagnoses and differential diagnoses.  The legacy rule-adjudication profile
remains available explicitly, but is not the default batch/CLI workflow.

See docs/ecg_agent_architecture.md for the design this implements.
"""
from __future__ import annotations

from .agent.diagnostic import ECGDiagnosticAgent, run_diagnostic_agent
from .agent.loop import AgentResult, ECGAgent, run_agent
from .backends import build_backend
from .evidence.store import EvidenceStore, EvidenceValue
from .evidence.briefing import build_chart_briefing
from .evidence.diagnostic_briefing import build_diagnostic_briefing
from .tools.registry import ToolRegistry, ToolResult, ToolSpec, build_default_registry
from .report import render_brief_report, render_human_report

__all__ = [
    "AgentResult",
    "ECGAgent",
    "ECGDiagnosticAgent",
    "EvidenceStore",
    "EvidenceValue",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_backend",
    "build_chart_briefing",
    "build_diagnostic_briefing",
    "build_default_registry",
    "run_agent",
    "run_diagnostic_agent",
    "render_brief_report",
    "render_human_report",
]
