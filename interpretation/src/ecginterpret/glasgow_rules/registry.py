from __future__ import annotations

from typing import Iterable, List

from .models import FIDELITIES, SOURCE_KINDS, RuleSpec


class RuleRegistry:
    def __init__(self, rules: Iterable[RuleSpec] = ()) -> None:
        self._rules: List[RuleSpec] = []
        self._ids: set[str] = set()
        for rule in rules:
            self.register(rule)

    @property
    def rules(self) -> List[RuleSpec]:
        return sorted(self._rules, key=lambda item: (item.order, item.rule_id))

    def register(self, rule: RuleSpec) -> None:
        if rule.rule_id in self._ids:
            raise ValueError(f"duplicate Glasgow rule id: {rule.rule_id}")
        if not rule.rule_id.startswith("GAN-"):
            raise ValueError(f"invalid Glasgow rule id: {rule.rule_id}")
        if not rule.chapter.strip():
            raise ValueError(f"chapter is required for {rule.rule_id}")
        if int(rule.pdf_page) <= 0:
            raise ValueError(f"pdf_page must be positive for {rule.rule_id}")
        if not rule.source_reference.strip():
            raise ValueError(f"source_reference is required for {rule.rule_id}")
        if rule.fidelity not in FIDELITIES:
            raise ValueError(f"invalid fidelity for {rule.rule_id}: {rule.fidelity}")
        if rule.source_kind not in SOURCE_KINDS:
            raise ValueError(f"invalid source kind for {rule.rule_id}: {rule.source_kind}")
        if rule.summary_code is not None and rule.summary_code not in {1, 2, 3, 4, 5, 6}:
            raise ValueError(f"invalid summary code for {rule.rule_id}: {rule.summary_code}")
        self._ids.add(rule.rule_id)
        self._rules.append(rule)
