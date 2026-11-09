"""Reference AI PCP arm: the LLM talks and parses, the engine decides.

Per call (one patient-visible message), stateless like Compass:
1. Router (LLM) picks one complaint script, or none (out of lane).
2. Extractors A and B (different vendors) code the conversation into findings
   for that script's features; the reconciler keeps only agreements.
3. Findings merge with the Clinical Context (none on HealthBench).
4. ``workup.step`` decides ask / defer / escalate / complete. The pipeline
   runs ``n_runs`` times; any disagreement in decision or asked features defers.
5. The dialogue LLM words the decision. Before physician sign-off (rung 0-1),
   patient messages give no individualized treatment advice.

This is the evaluation reference implementation. The production integration
in Compass (ENG-7) must reproduce the same decisions from the same inputs.
"""

from dataclasses import dataclass, field
from typing import Any

from compass_pcp.arms.baselines import EMERGENCY_REPLY, OUTPUT_LIMITS
from compass_pcp.arms.llm import LLM, parse_json
from compass_pcp.engine.scripts import IllnessScript
from compass_pcp.engine.workup import DecisionKind, WorkupConfig, WorkupState, step
from compass_pcp.findings import CodeSystem, Finding, Polarity, Provenance, reconcile

ROUTER = (
    "Pick the one complaint category that best matches the patient's main health concern. Categories:\n{cats}\n"
    'Reply JSON only: {{"category": "<id or none>", "suspected": ["..."]}}. Use "none" when no category fits or the '
    "request is not about a new symptom. suspected: up to 3 diagnoses you suspect, each either an id listed for the "
    "chosen category or \"other:<short name>\" if your suspicion is not listed."
)
EXTRACTOR = (
    "Code what the PATIENT has said in this conversation. For each feature id below, answer present, absent, or "
    "unknown. present/absent only when the patient clearly said so; anything unclear, hypothetical, or said only by "
    "the assistant is unknown. Mark confirmed=true only if the assistant explicitly asked about that feature and the "
    "patient explicitly answered.\nFeatures:\n{features}\n"
    'Reply JSON only: {{"<feature_id>": {{"value": "present|absent|unknown", "confirmed": true|false}}, ...}}'
)
ASK = (
    "Write the next text message to the patient asking these questions, warmly and simply, without suggesting a "
    "diagnosis. Questions:\n{qs}\n" + OUTPUT_LIMITS
)
HANDOFF = (
    "Write a short text to the patient: thank them, say a physician on the care team will review their messages "
    "{when} and reply, and tell them to call 911 or go to the ER if they get much worse. No diagnosis, no treatment "
    "advice. " + OUTPUT_LIMITS
)
ESCALATE = (
    "Write a short, calm, direct text telling the patient their symptoms need to be checked {where} today, and that "
    "you have alerted the on-call physician. Give the reason in plain words without naming a diagnosis. "
    + OUTPUT_LIMITS
)


@dataclass
class AiPcpArm:
    scripts: dict[str, IllnessScript]
    router: LLM
    extractor_a: LLM
    extractor_b: LLM | None  # None = single-extractor ablation
    dialogue: LLM
    config: WorkupConfig = field(default_factory=WorkupConfig)
    n_runs: int = 3

    @property
    def vendors(self) -> dict[str, str]:
        v = {"generator": self.dialogue.vendor, "router": self.router.vendor, "extractor_a": self.extractor_a.vendor}
        if self.extractor_b is not None:
            v["extractor_b"] = self.extractor_b.vendor
        return v

    def _route(self, messages) -> tuple[str | None, list[str]]:
        cats = "\n".join(
            f"- {k}: {s.complaint} (diagnoses: {', '.join(d.id for d in s.diagnoses)})" for k, s in self.scripts.items()
        )
        try:
            out = parse_json(self.router.complete(ROUTER.format(cats=cats), messages, max_tokens=200))
        except ValueError:
            return None, []
        cat = out.get("category")
        return (cat if cat in self.scripts else None), [str(x) for x in (out.get("suspected") or [])]

    def _extract(self, llm: LLM, script: IllnessScript, messages) -> list[Finding]:
        feats = "\n".join(f"- {f.id}: {f.question}" for f in script.features)
        try:
            out = parse_json(llm.complete(EXTRACTOR.format(features=feats), messages, max_tokens=900))
        except ValueError:
            return []
        found = []
        for fid, v in out.items():
            if fid not in {f.id for f in script.features} or not isinstance(v, dict):
                continue
            val = v.get("value")
            if val not in ("present", "absent"):
                continue
            found.append(
                Finding(
                    feature_id=fid, code_system=CodeSystem.LOCAL, code=fid,
                    polarity=Polarity.PRESENT if val == "present" else Polarity.ABSENT,
                    provenance=Provenance.PATIENT_REPORT, extractor_id=f"{llm.vendor}:{llm.model}",
                    confirmed=bool(v.get("confirmed")),
                )
            )
        return found

    def _one_run(self, messages, context_findings: list[Finding]):
        script_id, proposed = self._route(messages)
        if script_id is None:
            return None, None, {"route": None}
        script = self.scripts[script_id]
        a = self._extract(self.extractor_a, script, messages)
        if self.extractor_b is None:
            accepted, disputed = a, []
        else:
            r = reconcile(a, self._extract(self.extractor_b, script, messages))
            accepted, disputed = r.accepted, r.disputed
        turns = sum(1 for m in messages if m["role"] == "assistant")
        state = WorkupState(
            findings=[*context_findings, *accepted],
            unconfirmed_after_question=[d for d in disputed if turns > 0],
            turns_used=turns,
            llm_proposed_diagnoses=proposed,  # "other:..." entries are schema gaps and defer
        )
        return script, step(script, state, self.config), {"route": script_id, "disputed": disputed}

    def next_message(self, messages, *, configuration=None, context=None) -> tuple[str, dict[str, Any]]:
        context_findings: list[Finding] = list((context or {}).get("findings", []))
        runs = [self._one_run(messages, context_findings) for _ in range(self.n_runs)]
        signatures = {
            (None if d is None else (s.complaint, d.kind.value, tuple(sorted(d.ask_features)))) for s, d, _ in runs
        }
        script, decision, meta = runs[0]
        meta = {**meta, "arm": "ai_pcp", "n_runs": self.n_runs, "kappa_unanimous": len(signatures) == 1}
        if decision is not None:
            meta["certificate"] = decision.certificate.model_dump(mode="json")
        if len(signatures) > 1:
            meta["decision"] = "defer_run_disagreement"
            return self.dialogue.complete(HANDOFF.format(when="today"), messages), meta
        if decision is None:
            meta["decision"] = "defer_out_of_lane"
            return self.dialogue.complete(HANDOFF.format(when="today"), messages), meta
        meta["decision"] = decision.kind.value
        if decision.kind == DecisionKind.ESCALATE:
            return self.dialogue.complete(ESCALATE.format(where="in person at an ER or urgent care"), messages) or (
                EMERGENCY_REPLY
            ), meta
        if decision.kind == DecisionKind.ASK:
            qs = "\n".join(f"- {script.feature(f).question}" for f in decision.ask_features)
            return self.dialogue.complete(ASK.format(qs=qs), messages), meta
        if decision.kind == DecisionKind.DEFER:
            return self.dialogue.complete(HANDOFF.format(when="today"), messages), meta
        # COMPLETE: a plan goes to the physician; the patient gets no individualized advice before sign-off.
        return self.dialogue.complete(HANDOFF.format(when="within a few hours"), messages), meta
