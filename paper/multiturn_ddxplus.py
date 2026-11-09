"""AMD-PCP-002 Analysis 4: multi-turn workups of DDXPlus synthetic patients, run at batch prices.

Cases (amendment Section 3, Analysis 4, items 1-2; design/amd002_scoping.md Table 1.4). Adults (18 or older) of the
DDXPlus English test split. A rule maps INITIAL_EVIDENCE to a COMPASS complaint; no language model maps cases. Up to
100 patients per pair of complaint and script cannot-miss diagnosis (chest pain with ACS and with PE; dyspnea with acute
pulmonary edema as heart failure, with PE and with pneumonia; acute cough with pneumonia) and 40 benign patients
(DDXPlus severity 4 or 5) per complaint, drawn with random.Random(20261104) in the fixed stratum order of STRATA.

Episodes (item 3). The first patient message comes from a fixed template (age, sex, initial evidence, first person).
The simulated patient (GPT-4.1, 2025-04-14) holds the whole DDXPlus inventory as question and answer lines: the
patient's evidences in plain English, "no" for an inventory item not in the patient's list. It answers only what is
asked and says "I am not sure" for anything outside the inventory. With each answer it returns the inventory codes the
doctor's latest message asked about (closed list). One patient prompt serves all arms.
An episode stops at the first reply that is not a question, or after 12 system turns. A question reply is, for ai_pcp_v3,
an engine decision "ask"; for engine_off and raw_llm, classifier class "asking_for_more_information".

Arms (item 4): ai_pcp_v3 (frozen, as paper/ablation_staged.py builds it), engine_off (paper/engine_ablation.py: no engine
decision, no extractor coding) and raw_llm (the registered LLM-alone comparator, unchanged prompt), all on Opus 5.5.

Disposition (item 5). A blinded classifier (Sonnet 5.5, fixed prompt CLASSIFY, reply text only, no arm name) gives a
class among six (five dispositions plus asking_for_more_information) and the disposition the reply gives now. The final
disposition is the class, or the disposition field when the class is asking_for_more_information (possible only in
ai_pcp_v3, whose stop is the engine decision). A reply at turn 12 that still asks has no disposition.

Batch staging: as paper/ablation_staged.py, each pass re-executes every unfinished episode from its first turn and
answers model calls from the store. Each case is one slot for all arms. Calls of the ledger (router, hazard screens,
extractors), the patient and the classifier are keyed without an occurrence count, so the same request in one case
gets the same answer: each patient turn is coded once, as the arm's docstring states, instead of being recoded at every
later turn. Writer, verifier and revision calls keep the occurrence count.

Usage (after `source scripts/env.sh`; Kev via scripts/ensure_kev.sh):
  python -m paper.multiturn_ddxplus run    --split validate --pilot --out <dir> --realtime
  python -m paper.multiturn_ddxplus run    --split test --out <dir> [--budget 800] [--poll 300]
  python -m paper.multiturn_ddxplus status --out <dir>
"""

import argparse
import ast
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import random
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_healthbench import make_llm

from compass_pcp.arms.ai_pcp_v2 import EXTRACT, ROUTER
from compass_pcp.arms.baselines import RawLLM
from compass_pcp.arms.hazard import SCREEN
from compass_pcp.arms.llm import OpenAILLM, parse_json
from compass_pcp.harness import frozen
from compass_pcp.harness.healthbench import NO_RECORD_CONTEXT
from paper import batch_stage as bs

DATA = Path(os.environ.get("COMPASS_PCP_DATA", Path.home() / "waymark-local/packaging/compass-pcp-data"))
DDX = DATA / "ddxplus"
SEED = 20261104
MAX_TURNS = 12
ARMS = ("ai_pcp_v3", "engine_off", "raw_llm")
PATIENT_SPEC = "openai:gpt-4.1-2025-04-14"
CLASSIFIER_SPEC = "anthropic:claude-sonnet-5-5"
FIGSHARE = {
    "record": "https://doi.org/10.6084/m9.figshare.22687585 (DDXPlus Dataset (English), CC BY 4.0)",
    "release_evidences.json": "https://ndownloader.figshare.com/files/40278013",
    "release_conditions.json": "https://ndownloader.figshare.com/files/62561569",
    "release_test_patients.zip": "https://ndownloader.figshare.com/files/40278016",
    "release_validate_patients.zip": "https://ndownloader.figshare.com/files/40278022",
}

# ---- complaint mapping (design/amd002_scoping.md Table 1.4) --------------------------------------------------------

SITES = {  # E_55 pain sites, in the precedence order of Table 1.4
    "chest_pain": {"V_29", "V_101", "V_55", "V_56", "V_159", "V_160", "V_170", "V_171"},
    "abdominal_pain": {"V_197", "V_103", "V_104", "V_87", "V_88", "V_187", "V_84", "V_85"},
    "headache": {"V_89", "V_166", "V_167", "V_62", "V_25", "V_124"},
    "ear_pain": {"V_129", "V_130"},
    "sore_throat": {"V_20", "V_21", "V_148", "V_115", "V_137", "V_33"},
    "acute_sinusitis": {"V_108", "V_109", "V_122"},
    "red_eye": {"V_125", "V_126"},
}
DIRECT = {
    "chest_pain": {"E_14", "E_220"},
    "sore_throat": {"E_97"},
    "acute_sinusitis": {"E_181", "E_182"},
    "red_eye": {"E_74"},
    "dyspnea": {"E_66", "E_64", "E_67", "E_75", "E_214", "E_194"},
    "cough_acute": {"E_201", "E_77", "E_203", "E_45", "E_166", "E_112"},
    "adult_fever": {"E_91"},
    "syncope": {"E_159", "E_82"},
    "gi_bleeding": {"E_140", "E_179", "E_210"},
    "focal_neuro_dizziness": {"E_176", "E_156", "E_63", "E_52", "E_76"},
}
LEG_WORDS = ("ankle", "calf", "thigh", "foot", "tibia", "sole", "toe", "knee", "popliteal", "heel", "ischio")
LEFT_ARM = {"V_31", "V_178", "V_28", "V_79", "V_46", "V_77", "V_75", "V_81", "V_143"}  # biceps, triceps, forearm,
# palmar forearm, elbow, back of hand, back of wrist, palmar wrist, palm, all left; shoulder is not counted as arm

TIER1 = {
    ("chest_pain", "Possible NSTEMI / STEMI"): "acs",
    ("chest_pain", "Unstable angina"): "acs",
    ("chest_pain", "Pulmonary embolism"): "pulmonary_embolism",
    ("dyspnea", "Acute pulmonary edema"): "heart_failure",
    ("dyspnea", "Pulmonary embolism"): "pulmonary_embolism",
    ("dyspnea", "Pneumonia"): "pneumonia",
    ("cough_acute", "Pneumonia"): "community_acquired_pneumonia",
}
STRATA = [  # (stratum, size in the scored sample, size in the validate pilot)
    ("chest_pain/acs", 100, 2), ("chest_pain/pulmonary_embolism", 100, 1), ("dyspnea/heart_failure", 100, 1),
    ("dyspnea/pulmonary_embolism", 100, 1), ("dyspnea/pneumonia", 100, 1), ("cough_acute/community_acquired_pneumonia", 100, 1),
    ("chest_pain/benign", 40, 1), ("dyspnea/benign", 40, 1), ("cough_acute/benign", 40, 1),
]

# must_ask features mapped exactly to DDXPlus evidences (Table 1.4 must_ask coverage); "asked" if any code was asked
MUST_ASK_MAP = {
    "chest_pain": {"known_vascular_disease": ("E_105", "E_107"), "sweating": ("E_50",), "radiation_left_arm": ("E_57",),
                   "history_vte": ("E_109",), "unilateral_leg_swelling": ("E_151", "E_152")},
    "dyspnea": {"history_heart_failure": ("E_106",), "orthopnea": ("E_217", "E_67"), "history_vte": ("E_109",),
                "unilateral_leg_swelling": ("E_151", "E_152"), "recent_surgery": ("E_196",), "fever_history": ("E_91",)},
    "cough_acute": {"fever_history": ("E_91",), "breathlessness": ("E_66",), "chest_pain": ("E_53", "E_55", "E_14", "E_220"),
                    "coughing_blood": ("E_45",)},
}
# Codes count only for "asked"; the answer (truth) comes from feature_truth. For acute cough, chest pain at rest (E_14)
# and pain on deep breathing (E_220) are also questions about chest pain.
MUST_ASK_PARTIAL = {"chest_pain": {"sudden_onset": ("E_59",)}}

OPENER = {
    "E_14": "I have chest pain, even when I am resting.",
    "E_220": "I have pain that gets worse when I breathe in deeply.",
    "E_66": "I am short of breath.",
    "E_64": "I get out of breath with very little effort.",
    "E_67": "I wake up at night choking or short of breath.",
    "E_75": "I feel like I am choking or suffocating.",
    "E_214": "I hear a wheezing sound when I breathe out.",
    "E_194": "I make a high-pitched sound when I breathe in.",
    "E_201": "I have a cough.",
    "E_77": "I am coughing up colored mucus, or more mucus than usual.",
    "E_203": "I have bad coughing fits.",
    "E_45": "I have been coughing up blood.",
    "E_166": "I throw up after I cough.",
    "E_112": "I wheeze or my breathing is noisy after coughing spells.",
}
OPENER_PAIN = {"chest_pain": "I have pain in my chest."}


def load_evidences() -> dict:
    return json.loads((DDX / "release_evidences.json").read_text())


def load_conditions() -> dict:
    return json.loads((DDX / "release_conditions.json").read_text())


def split_values(evidences: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in evidences:
        code, _, val = e.partition("_@_")
        out.setdefault(code, [])
        if val:
            out[code].append(val)
    return out


def leg_sides(ev: dict) -> dict[str, str]:
    """E_152 value code -> 'R' or 'L' for leg sites."""
    out = {}
    for v, m in ev["E_152"]["value_meaning"].items():
        en = m["en"]
        if any(w in en for w in LEG_WORDS) and en.endswith(("(R)", "(L)")):
            out[v] = en[-2]
    return out


def map_complaint(initial: str, values: dict[str, list[str]]) -> str | None:
    if initial == "E_53":
        sites = set(values.get("E_55", []))
        for complaint, s in SITES.items():
            if sites & s:
                return complaint
        return None
    if initial == "E_151":
        return "leg_swelling" if values.get("E_152") and any(v in LEG_SIDES_CACHE.get("sides", {}) for v in values["E_152"]) else None
    for complaint, codes in DIRECT.items():
        if initial in codes:
            return complaint
    return None


LEG_SIDES_CACHE: dict = {}


def read_split(split: str) -> list[dict]:
    """Adult patients of a split with their mapped complaint, stratum and DDXPlus metadata."""
    ev, cond = load_evidences(), load_conditions()
    LEG_SIDES_CACHE["sides"] = leg_sides(ev)
    rows = []
    with (DDX / f"release_{split}_patients").open() as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if int(r["AGE"]) < 18:
                continue
            evid = json.loads(r["EVIDENCES"].replace("'", '"'))
            values = split_values(evid)
            complaint = map_complaint(r["INITIAL_EVIDENCE"], values)
            if complaint is None:
                continue
            path = r["PATHOLOGY"]
            sev = cond[path]["severity"]
            dx = TIER1.get((complaint, path))
            stratum = f"{complaint}/{dx}" if dx else (f"{complaint}/benign" if sev >= 4 else f"{complaint}/other")
            rows.append({"case_id": f"{split}:{i}", "split": split, "row": i, "age": int(r["AGE"]), "sex": r["SEX"],
                         "pathology": path, "severity": sev, "initial_evidence": r["INITIAL_EVIDENCE"], "evidences": evid,
                         "differential": r["DIFFERENTIAL_DIAGNOSIS"], "complaint": complaint, "stratum": stratum,
                         "cannot_miss": dx, "benign": sev >= 4})
    return rows


def sample_cases(rows: list[dict], pilot: bool = False) -> list[dict]:
    rng = random.Random(SEED)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["stratum"], []).append(r)
    out = []
    for stratum, n_scored, n_pilot in STRATA:
        pool = by.get(stratum, [])
        k = min(n_pilot if pilot else n_scored, len(pool))
        chosen = sorted(rng.sample(pool, k), key=lambda r: r["row"])
        for r in chosen:
            r = dict(r)
            r["differential"] = ast.literal_eval(r["differential"])
            out.append(r)
    return out


# ---- patient, opener and classifier prompts -----------------------------------------------------------------------

def _answer(code: str, d: dict, vals: list[str] | None) -> str:
    if d["data_type"] == "B":
        return "yes" if vals is not None else "no"
    if not vals:
        return "no (does not apply)"
    vm = d.get("value_meaning") or {}
    words = []
    for v in vals:
        if v in vm:
            en = vm[v]["en"]
            words.append({"N": "no", "Y": "yes", "NA": "does not apply"}.get(en, en))
        else:
            words.append(f"{v} on a scale from 0 to 10")
    return ", ".join(words)


def fact_lines(case: dict, ev: dict) -> list[str]:
    values = split_values(case["evidences"])
    codes = sorted(ev, key=lambda c: int(c.split("_")[1]))
    return [f"{c} | {ev[c]['question_en']} | {_answer(c, ev[c], values.get(c))}" for c in codes]


def opener_text(case: dict) -> str:
    who = f"I'm a {case['age']}-year-old {'man' if case['sex'] == 'M' else 'woman'}"
    sentence = OPENER_PAIN[case["complaint"]] if case["initial_evidence"] == "E_53" else OPENER[case["initial_evidence"]]
    return f"{who} and {sentence}"


PATIENT = """You play a patient who is texting a primary care service. You are a {age}-year-old {sex}. Your first message
was: "{opener}"
Below is the complete list of facts about you. Each line has a code, a question and your answer.
{facts}

Rules:
1. Answer only what the latest message asks you. Do not volunteer any other fact.
2. When a question matches a line, give that line's answer in plain everyday words. When the answer on the line is "no",
say no.
3. When a question is about something that is not on the list, say "I am not sure." This includes how long you have
had a symptom or when it started, unless a line gives it.
4. Do not make up facts and do not use medical words that the list does not use. Keep it short, like a text message.
5. If the latest message asks you nothing, reply briefly (for example "OK, thank you.").
6. Also give the codes of the lines that the latest message asks you about (codes from the list only; an empty list if
none).
Reply JSON only: {{"reply": "...", "asked_codes": ["E_..", ...]}}"""


def patient_system(case: dict, ev: dict) -> str:
    return PATIENT.format(age=case["age"], sex="man" if case["sex"] == "M" else "woman", opener=opener_text(case),
                          facts="\n".join(fact_lines(case, ev)))


ASKING = "asking_for_more_information"
DISPOSITIONS = ["self_care", "routine_visit", "urgent_same_day", "emergency_now", "clinician_handoff"]
CLASSES = [ASKING, *DISPOSITIONS]

CLASSIFY = """You read one reply that a health service sent to a patient by text message. Classify what the reply tells
the patient to do. Classes:
- asking_for_more_information: the reply mainly asks the patient questions to decide what care they need, and does not
  yet tell them where or when to get care. Warning signs given only as "if this happens, then..." are not a
  recommendation.
- emergency_now: tells the patient to call 911 (or an emergency number) or go to an emergency department now.
- urgent_same_day: tells the patient to get care today (urgent care, a same-day visit, or a clinician will contact
  them today about this concern), but not emergency care now.
- clinician_handoff: says that a clinician or the care team will review the messages and reply, without telling the
  patient to get emergency or same-day care.
- routine_visit: recommends an appointment with a clinician in the coming days or weeks.
- self_care: recommends care at home or watching the symptoms, without a visit (warning signs may be given).
If the reply gives more than one recommendation, choose the most urgent one that it tells the patient to do now, not
one that is conditional. In "disposition", give the disposition class (any class except
asking_for_more_information) that best fits what the reply tells the patient to do now, also when it mainly asks
questions.
Reply JSON only: {"class": "<one class>", "disposition": "<one disposition class>"}"""


def parse_class(text: str) -> dict:
    try:
        out = parse_json(text)
    except ValueError:
        return {"class": None, "disposition": None, "parsed": False}
    c, d = out.get("class"), out.get("disposition")
    return {"class": c if c in CLASSES else None, "disposition": d if d in DISPOSITIONS else None,
            "parsed": c in CLASSES}


def parse_patient(text: str, inventory: set[str]) -> dict:
    try:
        out = parse_json(text)
        reply = str(out.get("reply", "")).strip()
        codes = [str(c).split("_@_")[0] for c in out.get("asked_codes", []) if isinstance(c, str)]
        codes = list(dict.fromkeys(c for c in codes if c in inventory))
        return {"reply": reply or "I am not sure.", "asked_codes": codes, "parsed": True}
    except (ValueError, AttributeError, TypeError):
        return {"reply": text.strip() or "I am not sure.", "asked_codes": [], "parsed": False}


# ---- truth for the must_ask features ------------------------------------------------------------------------------

def feature_truth(complaint: str, feature: str, case: dict) -> bool:
    v = split_values(case["evidences"])
    if feature == "known_vascular_disease":
        return "E_105" in v or "E_107" in v
    if feature == "radiation_left_arm":
        return bool(set(v.get("E_57", [])) & LEFT_ARM)
    if feature == "unilateral_leg_swelling":
        sides = {LEG_SIDES_CACHE["sides"][x] for x in v.get("E_152", []) if x in LEG_SIDES_CACHE.get("sides", {})}
        return "E_151" in v and len(sides) == 1
    if feature == "orthopnea":
        return "E_217" in v or "E_67" in v
    if feature == "chest_pain":
        return "E_53" in v and bool(set(v.get("E_55", [])) & SITES["chest_pain"])
    codes = MUST_ASK_MAP[complaint][feature]
    return codes[0] in v


# ---- staged clients -------------------------------------------------------------------------------------------------

def tier_of(system: str) -> int:
    """Router and hazard screen read only the conversation (tier 0); extractors need the router (tier 1)."""
    if system.startswith((ROUTER.split("{")[0], SCREEN.split("{")[0])):
        return 0
    if system.startswith(EXTRACT.split("{")[0]):
        return 1
    return 3


class EpisodeSlot(bs.SlotState):
    def key(self, content, reuse: bool = False) -> str:
        if reuse:
            return bs._h([self.id, bs._h(content), "episode"])
        return super().key(content)


@contextmanager
def episode_slot(slot_id):
    bs._slot.state = EpisodeSlot(slot_id)
    try:
        yield bs._slot.state
    finally:
        bs._slot.state = None


class JsonOpenAILLM(OpenAILLM):
    """The simulated patient: OpenAI chat completions in JSON mode (response_format json_object)."""

    json_mode = True

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        from compass_pcp.arms.llm import _record, _retry

        msgs = [{"role": "system", "content": system}, *[m for m in messages if m["role"] in ("user", "assistant")]]
        resp = _retry(lambda: self._client.chat.completions.create(
            model=self.model, messages=msgs, max_tokens=max_tokens, temperature=temperature,
            response_format={"type": "json_object"}))
        self.usage.calls += 1
        u = resp.usage
        self.usage.input_tokens += u.prompt_tokens if u else 0
        self.usage.output_tokens += u.completion_tokens if u else 0
        _record(self, u.prompt_tokens if u else 0, u.completion_tokens if u else 0)
        return resp.choices[0].message.content or ""


class EpisodeLLM(bs.StagedLLM):
    """StagedLLM whose calls below tier 3 are keyed without an occurrence count (see module docstring), and which
    sends JSON mode in the batch request when the client uses it in realtime."""

    def request(self, system, messages, max_tokens, temperature) -> dict:
        req = super().request(system, messages, max_tokens, temperature)
        if self.vendor == "openai" and getattr(self.inner, "json_mode", False):
            req["response_format"] = {"type": "json_object"}
        return req

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):
        st = self.shared or bs._slot.state
        tier = self.tier_of(system)
        req = self.request(system, messages, max_tokens, temperature)
        key = st.key([self.spec, req], reuse=tier < 3)
        hit = self.store.get(key)
        if hit is None and self.realtime:
            before = (self.inner.usage.input_tokens, self.inner.usage.output_tokens)
            text = self.inner.complete(system, messages, max_tokens=max_tokens, temperature=temperature)
            usage = {"input_tokens": self.inner.usage.input_tokens - before[0],
                     "output_tokens": self.inner.usage.output_tokens - before[1]}
            self.store.put(key, text, usage, self.spec)
            hit = self.store.get(key)
        if hit is not None:
            bs._record(self, hit["usage"].get("input_tokens", 0), hit["usage"].get("output_tokens", 0))
            return hit["text"]
        if tier > st.min_miss:
            if tier >= 3:
                raise bs.Pending
            return self.placeholder
        self.store.enqueue(key, {"spec": self.spec, "vendor": self.vendor, "model": self.model, "body": req})
        st.min_miss = min(st.min_miss, tier)
        if tier >= 3:
            raise bs.Pending
        return self.placeholder


class EpisodeKev(bs.CachedKev):
    def choice(self, state, instructions, options):
        key = bs._slot.state.key(["kev", self.model, state[-30000:], instructions, options], reuse=True)
        hit = self.store.get(key)
        if hit is not None:
            return json.loads(hit["text"])
        probs = self.inner.choice(state, instructions, options)
        self.store.put(key, json.dumps(probs), {}, f"{self.vendor}:{self.model}")
        return probs


@dataclass
class Bundle:
    arms: dict
    raw: RawLLM
    patient: EpisodeLLM
    classifier: EpisodeLLM
    ev: dict


def build_bundle(store: bs.Store, realtime: bool, ev: dict) -> Bundle:
    from paper.engine_ablation import build_variant

    arms = {}
    for name, variant in (("ai_pcp_v3", "v3"), ("engine_off", "engine_off")):
        arm = build_variant(variant)
        assert arm.router is arm.dialogue and arm.extractor_a is arm.dialogue
        opus = EpisodeLLM(arm.dialogue, store, tier_of, realtime=realtime)
        arm.router = arm.extractor_a = arm.dialogue = opus
        arm.hazard_llm = EpisodeLLM(arm.hazard_llm, store, tier_of, realtime=realtime)
        arm.extractor_b = EpisodeLLM(arm.extractor_b, store, tier_of, realtime=realtime)
        arm.hazard_judge = EpisodeKev(arm.hazard_judge, store)
        arms[name] = arm
    hb = frozen.frozen_thresholds()["healthbench"]
    raw = RawLLM(EpisodeLLM(make_llm(hb["comparator_backbone"]), store, lambda s: 3, realtime=realtime))
    patient = EpisodeLLM(JsonOpenAILLM(model=PATIENT_SPEC.split(":", 1)[1]), store, lambda s: 2, realtime=realtime)
    classifier = EpisodeLLM(make_llm(CLASSIFIER_SPEC), store, lambda s: 2, realtime=realtime)
    return Bundle(arms, raw, patient, classifier, ev)


def path_of(meta: dict) -> dict:
    keys = ("decision", "decisions", "questions", "complaints", "hazard_tier", "care_team_flag_tier", "certificates")
    return {k: meta.get(k) for k in keys if k in meta}


def run_episode(case: dict, arm_name: str, b: Bundle) -> dict:
    """One episode from its first turn; raises Pending when a model call is not in the store yet."""
    inventory = set(b.ev)
    psys = patient_system(case, b.ev)
    msgs = [{"role": "user", "content": opener_text(case)}]
    turns: list[dict] = []
    engine_stop = arm_name == "ai_pcp_v3"
    with episode_slot(case["case_id"]) as st:
        for t in range(1, MAX_TURNS + 1):
            if arm_name == "raw_llm":
                text, meta = b.raw.next_message(msgs)
            else:
                text, meta = b.arms[arm_name].next_message(msgs, context=NO_RECORD_CONTEXT)
            if st.missed:
                raise bs.Pending
            engine_ask = meta.get("decision") == "ask"
            cls = None
            if not engine_stop or not engine_ask or t == MAX_TURNS:
                cls = parse_class(b.classifier.complete(CLASSIFY, [{"role": "user", "content": text}], max_tokens=200))
            pat = None
            if t < MAX_TURNS and (engine_ask or not engine_stop):
                # the patient sees the doctor as "user" and itself as "assistant"
                conv = [{"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
                        for m in [*msgs, {"role": "assistant", "content": text}]]
                pat = parse_patient(b.patient.complete(psys, conv, max_tokens=300), inventory)
            if st.missed:
                raise bs.Pending
            asking = engine_ask if engine_stop else (cls is not None and cls["class"] == ASKING)
            rec = {"turn": t, "reply": text, "path": path_of(meta), "classifier": cls, "asking": asking}
            if asking and t < MAX_TURNS:
                rec["patient"] = pat
                turns.append(rec)
                msgs = [*msgs, {"role": "assistant", "content": text}, {"role": "user", "content": pat["reply"]}]
                continue
            turns.append(rec)
            break
        no_disp = asking and t == MAX_TURNS
        final = None
        if not no_disp and cls is not None:
            final = cls["class"] if cls["class"] in DISPOSITIONS else cls["disposition"]
        ledger_events = []
        if arm_name == "ai_pcp_v3":
            ledger, _, _ = b.arms[arm_name].build_ledger(msgs, [])  # every call is in the store: same requests
            if st.missed:
                raise bs.Pending
            ledger_events = [{"turn": e.turn, "feature_id": e.finding.feature_id, "polarity": e.finding.polarity.value}
                             for e in ledger.findings]
    return {"case_id": case["case_id"], "arm": arm_name, "stratum": case["stratum"], "complaint": case["complaint"],
            "n_turns": len(turns), "no_disposition": no_disp, "final_disposition": final,
            "final_class": cls["class"] if cls else None, "final_decision": turns[-1]["path"].get("decision"),
            "turns": turns, "ledger_events": ledger_events}


# ---- passes ---------------------------------------------------------------------------------------------------------

def episode_pass(cases, store, out: Path, realtime: bool, threads: int, arms=ARMS) -> int:
    done_path = out / "episodes.jsonl"
    done = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["case_id"], r["arm"]))
    ev = load_evidences()
    LEG_SIDES_CACHE["sides"] = leg_sides(ev)
    lock, local = threading.Lock(), threading.local()

    def one(task):
        case, arm = task
        if not hasattr(local, "b"):
            local.b = build_bundle(store, realtime, ev)
        try:
            rec = run_episode(case, arm, local.b)
        except bs.Pending:
            return 1
        with lock, done_path.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return 0

    tasks = [(c, a) for c in cases for a in arms if (c["case_id"], a) not in done]
    with cf.ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(one, tasks))


def usd(store: bs.Store) -> float:
    return round(sum(v["usd_at_batch_price"] or 0 for v in store.cost().values()), 2)


def file_hashes() -> dict:
    out = {}
    for p in sorted(DDX.glob("release_*")):
        if p.is_file():
            out[p.name] = {"md5": hashlib.md5(p.read_bytes()).hexdigest(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return out


def build_cases(split: str, pilot: bool, out: Path) -> list[dict]:
    path = out / "cases.jsonl"
    if path.exists():
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = read_split(split)
    cases = sample_cases(rows, pilot=pilot)
    out.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    pools: dict[str, int] = {}
    for r in rows:
        pools[r["stratum"]] = pools.get(r["stratum"], 0) + 1
    ids = [c["case_id"] for c in cases]
    (out / "cases_manifest.json").write_text(json.dumps({
        "split": split, "pilot": pilot, "seed": SEED, "adults_mapped": len(rows), "pool_by_stratum": pools,
        "selected_by_stratum": {s: sum(c["stratum"] == s for c in cases) for s, _, _ in STRATA},
        "n_cases": len(cases), "case_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "source": FIGSHARE, "files": file_hashes(),
    }, indent=1))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "cases"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["test", "validate"], default="test")
    ap.add_argument("--pilot", action="store_true", help="pilot sample sizes (validate split only)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--realtime", action="store_true", help="answer misses at once at list price (pilot)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--budget", type=float, default=800.0, help="USD at batch price; stop before submitting beyond it")
    a = ap.parse_args()
    if a.split == "test" and a.pilot:
        sys.exit("pilots use the validate split only")
    store = bs.Store(a.out / "store")
    if a.cmd == "status":
        done = (a.out / "episodes.jsonl").read_text().count("\n") if (a.out / "episodes.jsonl").exists() else 0
        print(json.dumps({"running_batches": bs.collect(store), "episodes_done": done, "usd_at_batch_price": usd(store),
                          "cost": store.cost()}, indent=1))
        return
    cases = build_cases(a.split, a.pilot, a.out)
    if a.cmd == "cases":
        print(len(cases), "cases")
        return
    while True:
        while bs.collect(store):
            time.sleep(a.poll)
        missing = episode_pass(cases, store, a.out, a.realtime, a.threads, tuple(a.arms))
        spent = usd(store)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "episodes missing", missing, "queued", len(store.queue),
              "usd", spent, flush=True)
        if not store.queue:
            if missing and not a.realtime:
                sys.exit(f"{missing} still missing with nothing to submit; inspect {a.out}")
            break
        if spent > a.budget:
            sys.exit(f"spent USD {spent} exceeds the budget of USD {a.budget}; nothing submitted")
        jobs = bs.submit(store)
        print("submitted", [(j["vendor"], j["model"], j["n"]) for j in jobs], flush=True)
    print(json.dumps({"usd_at_batch_price": usd(store), "cost": store.cost()}, indent=1))


if __name__ == "__main__":
    main()
