"""Dual-process arm: a fast tier on every turn, a deliberate tier only when a trigger fires.

The fast tier is the AI PCP pipeline on a fast model. It builds the ledger and
the engine decision; if any trigger fires, the deliberate tier (the same
pipeline on a reasoning model, with repeated runs) produces the reply instead.
Triggers can only raise scrutiny: the fast tier never answers a turn that a
trigger marks for deliberation. Trigger sensitivity is measured against the
cases where the deliberate tier's reply differs materially (see harness).
"""

from dataclasses import dataclass
from typing import Any

from compass_pcp.arms.ai_pcp_v2 import AiPcpV2
from compass_pcp.engine.deferral import DeferralReason
from compass_pcp.engine.stopping import CannotMissStatus
from compass_pcp.ledger import HazardTier

TRIGGER_REASONS = {
    DeferralReason.SCHEMA_GAP,
    DeferralReason.CONTRADICTION,
    DeferralReason.UNCONFIRMED_DECISION_FINDING,
    DeferralReason.RUN_DISAGREEMENT,
}


@dataclass
class DualProcess:
    fast: AiPcpV2
    deliberate: AiPcpV2

    def triggers(self, messages, context) -> tuple[list[str], dict, tuple]:
        built = self.fast.build_ledger(messages, list((context or {}).get("findings", [])))
        ledger, suspected, disputed = built
        turns = sum(1 for m in messages if m["role"] == "user")
        decisions, _ = self.fast._decide(ledger, suspected, disputed, turns)
        fired = []
        if self.fast.user_type == "health_professional":
            fired.append("clinician_request")
        if ledger.hazard_tier >= HazardTier.URGENT_SAME_DAY:
            fired.append(f"hazard_{ledger.hazard_tier.name.lower()}")
        if disputed:
            fired.append("extractor_disagreement")
        for _, d in decisions:
            if any(c.status == CannotMissStatus.NEEDS_WORKUP for c in d.certificate.cannot_miss):
                fired.append("cannot_miss_middle_zone")
            if TRIGGER_REASONS & set(d.certificate.deferral_reasons):
                fired.append("engine_uncertainty")
        return sorted(set(fired)), {"hazard_tier": ledger.hazard_tier.name, "complaints": ledger.active_complaints}, built

    def next_message(self, messages, *, configuration=None, context=None) -> tuple[str, dict[str, Any]]:
        fired, info, built = self.triggers(messages, context)
        if fired:
            text, meta = self.deliberate.next_message(messages, configuration=configuration, context=context)
            return text, {**meta, "arm": "dual_process", "tier": "deliberate", "triggers": fired, **info}
        text, meta = self.fast.next_message(messages, configuration=configuration, context=context, prebuilt=built)
        return text, {**meta, "arm": "dual_process", "tier": "fast", "triggers": [], **info}
