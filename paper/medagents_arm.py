"""MedAgents (Tang et al., Findings of ACL 2024) on HealthBench Consensus: AMD-PCP-002 Analysis 6.

Port of the "syn_verif" pipeline in github.com/gersteinlab/MedicalAgentsBench, baselines/MedAgents (commit
2e0ee95748523e02d15051f9fbe2961538afdd92, MIT License, Copyright (c) 2025 Gerstein Lab; the full notice is in
design/medagents_fidelity.md). Prompt builders and cleansing functions are copied verbatim from prompt_utils.py and
data_utils.py. The per-step max_tokens caps come from the authors' original repository (gersteinlab/MedAgents,
commit aaeff04, utils.py). The changes are D1 to D6 in design/amd002_scoping.md section 2.2; every departure is listed
in design/medagents_fidelity.md:
  D1  Claude Opus 5.5 in every role. Opus 5.5 rejects `temperature` and cannot turn thinking off, so every call sends
      no temperature and output_config.effort = "low" (the lowest setting). Thinking counts against max_tokens, so
      each call gets the 2,048-token thinking headroom of the frozen AnthropicLLM, and the visible text is then cut to
      the original cap (CHARS_PER_TOKEN characters per token), as the original API cap would have cut it.
  D2  the conversation, as a role-tagged transcript, fills the question slot.
  D3  options = '' (the authors' MedicationQA path).
  D4  the final prompt asks for the reply to the user's last message under OUTPUT_LIMITS, the output policy shared by
      every HealthBench arm, with the shared reply budget MAX_TOKENS.
  D5  original caps: 50 per domain classification, 300 per analysis, 2,500 for synthesis and revision, 30 per vote,
      500 per advice.
  D6  everything else unchanged: 5 question domains, 2 option domains, up to 3 vote rounds, the "General Medicine"
      fallback, and the option-instruction repetition inside the expert loop of get_options_analysis_prompt.
Calls are stateless. Within a stage, independent calls (domains; question analyses; option analyses; votes; advice)
are issued together, so one reply needs at most 14 batch stages. The order of independent calls does not change any
prompt, because no call in a stage reads another call of the same stage.

Usage (after `source scripts/env.sh`):
  python -m paper.medagents_arm run    --data <consensus_*.jsonl> --out <dir> [--n 5] [--realtime]
  python -m paper.medagents_arm status --out <dir>
  python -m paper.medagents_arm analyze --out <dir> --registered <test_healthbench dir>
`run` loops sample passes and batch submissions until every reply exists, then grades with both frozen graders
through the same HealthBenchEval pass as paper/ablation_staged.py. It is safe to stop and restart at any point.
"""

import argparse
import ast
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
from pathlib import Path
from string import punctuation

from compass_pcp.arms.baselines import MAX_TOKENS, OUTPUT_LIMITS
from paper.batch_stage import Pending, Store, _slot, collect, slot, submit

MODEL = "claude-opus-5-5"
SPEC = f"anthropic:{MODEL}"
EFFORT = "low"
THINKING_HEADROOM = 2048  # as compass_pcp.arms.llm.AnthropicLLM; the final reply always uses it
CAPPED_HEADROOM = 2048  # thinking headroom on the capped steps; --headroom sets it (see design/medagents_fidelity.md)
CHARS_PER_TOKEN = 2.9  # median 2.92 for Opus 5.5 on 25 HealthBench ideal completions (count_tokens, 2026-11-03)
ARM = "medagents"
NLTK_DATA = Path(__file__).resolve().parents[2] / "compass-pcp-data" / "vendor" / "nltk_data"

# Original caps (gersteinlab/MedAgents utils.py, commit aaeff04).
CAP_DOMAIN, CAP_ANALYSIS, CAP_SYNTH, CAP_VOTE, CAP_ADVICE, CAP_REVISION = 50, 300, 2500, 30, 500, 2500
MAX_ATTEMPT_VOTE = 3  # main.py default

# ---- verbatim from MedicalAgentsBench baselines/MedAgents/prompt_utils.py -----------------------------------------

NUM_QD = 5
NUM_OD = 2


def get_question_domains_prompt(question):
    question_domain_format = "Medical Field: " + " | ".join(["Field" + str(i) for i in range(NUM_QD)])
    question_classifier = "You are a medical expert who specializes in categorizing a specific medical scenario into specific areas of medicine."
    prompt_get_question_domain = f"You need to complete the following steps:" \
            f"1. Carefully read the medical scenario presented in the question: '''{question}'''. \n" \
            f"2. Based on the medical scenario in it, classify the question into five different subfields of medicine. \n" \
            f"3. You should only output in exactly the same format as '''{question_domain_format}'''."
    return question_classifier, prompt_get_question_domain


def get_question_analysis_prompt(question, question_domain):
    question_analyzer = f"You are a medical expert in the domain of {question_domain}. " \
        f"From your area of specialization, you will scrutinize and diagnose the symptoms presented by patients in specific medical scenarios."
    prompt_get_question_analysis = f"Please meticulously examine the medical scenario outlined in this question: '''{question}'''." \
                        f"Drawing upon your medical expertise, interpret the condition being depicted. " \
                        f"Subsequently, identify and highlight the aspects of the issue that you find most alarming or noteworthy."

    return question_analyzer, prompt_get_question_analysis


def get_options_domains_prompt(question, options):
    options_domain_format =  "Medical Field: " + " | ".join(["Field" + str(i) for i in range(NUM_OD)])
    options_classifier = f"As a medical expert, you possess the ability to discern the two most relevant fields of expertise needed to address a multiple-choice question encapsulating a specific medical context."
    prompt_get_options_domain = f"You need to complete the following steps:" \
                f"1. Carefully read the medical scenario presented in the question: '''{question}'''." \
                f"2. The available options are: '''{options}'''. Strive to understand the fundamental connections between the question and the options." \
                f"3. Your core aim should be to categorize the options into two distinct subfields of medicine. " \
                f"You should only output in exactly the same format as '''{options_domain_format}'''"
    return options_classifier, prompt_get_options_domain


def get_options_analysis_prompt(question, options, op_domain, question_analysis):
    option_analyzer = f"You are a medical expert specialized in the {op_domain} domain. " \
                f"You are adept at comprehending the nexus between questions and choices in multiple-choice exams and determining their validity. " \
                f"Your task, in particular, is to analyze individual options with your expert medical knowledge and evaluate their relevancy and correctness."

    prompt_get_options_analyses = f"Regarding the question: '''{question}''', we procured the analysis of five experts from diverse domains. \n"
    for _domain, _analysis in question_analysis.items():
        prompt_get_options_analyses += f"The evaluation from the {_domain} expert suggests: {_analysis} \n"
        prompt_get_options_analyses += f"The following are the options available: '''{options}'''." \
                    f"Reviewing the question's analysis from the expert team, you're required to fathom the connection between the options and the question from the perspective of your respective domain, " \
                    f"and scrutinize each option individually to assess whether it is plausible or should be eliminated based on reason and logic. "\
                    f"Pay close attention to discerning the disparities among the different options and rationalize their existence. " \
                    f"A handful of these options might seem right on the first glance but could potentially be misleading in reality."
    return option_analyzer, prompt_get_options_analyses


def get_synthesized_report_prompt(question_analyses, option_analyses):
    synthesizer = "You are a medical decision maker who excels at summarizing and synthesizing based on multiple experts from various domain experts."

    syn_report_format = f"Key Knowledge: [extracted key knowledge] \n" \
                f"Total Analysis: [synthesized analysis] \n"
    prompt = f"Here are some reports from different medical domain experts.\n "
    prompt += f"You need to complete the following steps:" \
                f"1. Take careful and comprehensive consideration of the following reports." \
                f"2. Extract key knowledge from the following reports. " \
                f"3. Derive the comprehensive and summarized analysis based on the knowledge." \
                f"4. Your ultimate goal is to derive a refined and synthesized report based on the following reports." \
                f"You should output in exactly the same format as '''{syn_report_format}'''"
    prompt += question_analyses
    prompt += option_analyses

    return synthesizer, prompt


def get_consensus_prompt(domain, syn_report):
    voter = f"You are a medical expert specialized in the {domain} domain."
    cons_prompt = f"Here is a medical report: {syn_report} \n"\
        f"As a medical expert specialized in {domain}, please carefully read the report and decide whether your opinions are consistent with this report." \
        f"Please respond only with: [YES or NO]."
    return voter, cons_prompt


def get_consensus_opinion_prompt(domain, syn_report):
    opinion_prompt = f"Here is a medical report: {syn_report} \n"\
        f"As a medical expert specialized in {domain}, please make full use of your expertise to propose revisions to this report." \
        f"You should output in exactly the same format as '''Revisions: [proposed revision advice] '''"
    return opinion_prompt


def get_revision_prompt(syn_report, revision_advice):
    revision_prompt = f"Here is the original report: {syn_report}\n\n"
    for domain, advice in revision_advice.items():
        revision_prompt += f"Here is advice from a medical expert specialized in {domain}: {advice}.\n"
    revision_prompt += f"Based on the above advice, output the revised analysis in exactly the same format as '''Total Analysis: [revised analysis] '''"
    return revision_prompt


# ---- D4: the one changed prompt. Original get_final_answer_prompt_wsyn: "Here is a synthesized report: {syn_report} \n
# Based on the above report, select the optimal choice to answer the question. \nPoints to note: \n1. The analyses
# provided should guide you towards the correct response. \n2. Any option containing incorrect information inherently
# cannot be the correct choice. \n3. Please respond only with the selected option's letter ..." -------------------------

def get_final_reply_prompt_wsyn(syn_report):
    prompt = f"Here is a synthesized report: {syn_report} \n" \
        f"Based on the above report, write the reply to the user's last message in the conversation. \n" \
        f"Points to note: \n" \
        f"1. The analyses provided should guide you towards the correct response. \n" \
        f"2. {OUTPUT_LIMITS}"
    return prompt


# ---- verbatim from MedicalAgentsBench baselines/MedAgents/data_utils.py --------------------------------------------

def sent_tokenize(text):
    import nltk

    if str(NLTK_DATA) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA))
    return nltk.tokenize.sent_tokenize(text)


def remove_incomplete_sentence(text):
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and sentences[-1][-1] != '.':
        return ' '.join(sentences[:-1]) + '.'   #remove the last sentence
    else:
        return text


def cleansing_analysis(analyses, domains, type):
    analysis = {}

    for i, item in enumerate(analyses):
        if item == "ERROR.":
            item = f"There is no analysis for this {type}."
        item = remove_incomplete_sentence(item)
        if "as an ai language model" in item.lower():
            end_index = item.lower().find("as an ai language model")+len("as an ai language model")
            item= item[end_index:].strip().strip(',').strip()
        analysis[domains[i]] = item

    return analysis


def cleansing_syn_report(question, options, raw_synthesized_report):

    tmp = raw_synthesized_report.split("Total Analysis:")
    try:
        total_analysis_text = tmp[1].strip()
    except:  # noqa: E722 (verbatim)
        total_analysis_text = tmp[0].strip()
    if "Key Knowledge" in tmp:
        key_knowledge_text = tmp[0].split("Key Knowledge:")[-1].strip()
        final_syn_repo = f"Question: {question} \n" \
            f"Options: {options} \n" \
            f"Key Knowledge: {key_knowledge_text} \n" \
            f"Total Analysis: {total_analysis_text} \n"
    else:
        final_syn_repo = f"Question: {question} \n" \
            f"Options: {options} \n" \
            f"Total Analysis: {total_analysis_text} \n"

    return final_syn_repo


def cleansing_voting(output):
    import re

    output = output.lower()
    ans = re.findall(r'yes|no', output)
    if len(ans) == 0:
        ans = "yes"
    else:
        ans = ans[0]
    return ans


def transform_dict2text(analyses, type, content):
    if type == "question":
        report = ""
        i = 0
        for _domain, _analysis in analyses.items():
            report += f"Report{i} \n" \
                f"Question: {content} \n" \
                f"Domain: {_domain} \n" \
                f"Analysis: {_analysis} \n\n"
            i += 1
    elif type == "options":
        report = ""
        i = 0
        for _domain, _analysis in analyses.items():
            report += f"Report{i}: \n" \
                f"Options: {content} \n" \
                f"Domain: {_domain} \n" \
                f"Analysis: {_analysis} \n\n"
            i += 1
    return report


# ---- D2: the conversation fills the question slot ------------------------------------------------------------------

def render_question(messages) -> str:
    """Role-tagged transcript, then main.py's rule: add '?' when the text does not end in punctuation."""
    q = "\n\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages
                    if m["role"] in ("user", "assistant")).strip()
    return q if q and q[-1] in punctuation else q + '?'


# ---- D1: the Opus 5.5 adapter for api_handler.get_output_multiagent ------------------------------------------------

def request_body(system_role: str, user_input: str, max_tokens: int, headroom: int = THINKING_HEADROOM) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": user_input}],
            "max_tokens": max_tokens + headroom, "output_config": {"effort": EFFORT}}
    if system_role:  # the original sends an empty system message for system_role=""; omitting it is equivalent
        body["system"] = system_role
    return body


def cut_to_cap(text: str, cap: int | None) -> str:
    """The original API stopped visible text at `cap` tokens; cut at the calibrated character equivalent."""
    return text if cap is None else text[: int(cap * CHARS_PER_TOKEN)]


class Opus:
    """Staged (batch) or realtime calls; every result goes to the Store under the batch_stage slot key."""

    def __init__(self, store: Store, realtime: bool = False):
        self.store, self.realtime = store, realtime
        self._client = None
        self._local = threading.local()

    def usage(self) -> dict:
        return self._local.__dict__.setdefault("u", {"calls": 0, "input_tokens": 0, "output_tokens": 0})

    def reset_usage(self) -> None:
        self._local.u = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def _record(self, usage: dict) -> None:
        u = self.usage()
        u["calls"] += 1
        u["input_tokens"] += usage.get("input_tokens", 0)
        u["output_tokens"] += usage.get("output_tokens", 0)

    def lookup(self, system_role, user_input, max_tokens, headroom=THINKING_HEADROOM):
        """Text of one call (raw, before the cap cut) or Pending after queueing it."""
        body = request_body(system_role, user_input, max_tokens, headroom)
        key = _slot.state.key([SPEC, body])
        hit = self.store.get(key)
        if hit is not None:
            self._record(hit["usage"])
            return hit["text"]
        if self.realtime:
            import anthropic

            if self._client is None:
                self._client = anthropic.Anthropic(timeout=600.0, max_retries=4)
            r = self._client.messages.create(**body)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
            self.store.put(key, text, usage, SPEC)
            self._record(usage)
            return text
        self.store.enqueue(key, {"spec": SPEC, "vendor": "anthropic", "model": MODEL, "body": body})
        raise Pending

    def stage(self, calls: list[tuple[str, str, int | None]]) -> list[str]:
        """One stage of independent calls (system_role, user_input, cap; cap None = MAX_TOKENS, no cut). Every miss
        is queued before Pending is raised, so the whole stage goes into one batch."""
        out, missing = [], False
        for system_role, user_input, cap in calls:
            try:
                text = (self.lookup(system_role, user_input, MAX_TOKENS) if cap is None
                        else self.lookup(system_role, user_input, cap, CAPPED_HEADROOM))
            except Pending:
                missing = True
                continue
            # get_output_multiagent returns "ERROR." when the API gives no content; an empty text block is the same
            out.append(cut_to_cap(text, cap) if text.strip() else "ERROR.")
        if missing:
            raise Pending
        return out


# ---- the syn_verif pipeline (MedicalAgentsBench utils.fully_decode), staged ----------------------------------------

def fully_decode(messages, llm: Opus, max_attempt_vote: int = MAX_ATTEMPT_VOTE) -> tuple[str, dict]:
    question = render_question(messages)
    options = ''  # D3

    # get question domains; get option domains (stage 1)
    question_classifier, prompt_get_question_domain = get_question_domains_prompt(question)
    options_classifier, prompt_get_options_domain = get_options_domains_prompt(question, options)
    raw_question_domain, raw_option_domain = llm.stage([
        (question_classifier, prompt_get_question_domain, CAP_DOMAIN),
        (options_classifier, prompt_get_options_domain, CAP_DOMAIN)])
    if raw_question_domain == "ERROR.":
        raw_question_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_QD)])
    question_domains = [domain.strip() for domain in raw_question_domain.split("Field:")[-1].strip().split("|")]
    if raw_option_domain == "ERROR.":
        raw_option_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_OD)])
    options_domains = [domain.strip() for domain in raw_option_domain.split("Field:")[-1].strip().split("|")]

    # get question analysis (stage 2)
    tmp_question_analysis = llm.stage([(*get_question_analysis_prompt(question, d), CAP_ANALYSIS) for d in question_domains])
    question_analyses = cleansing_analysis(tmp_question_analysis, question_domains, 'question')

    # get option analysis (stage 3)
    tmp_option_analysis = llm.stage([(*get_options_analysis_prompt(question, options, d, question_analyses), CAP_ANALYSIS)
                                     for d in options_domains])
    option_analyses = cleansing_analysis(tmp_option_analysis, options_domains, 'option')

    # get synthesized report (stage 4)
    q_analyses_text = transform_dict2text(question_analyses, "question", question)
    o_analyses_text = transform_dict2text(option_analyses, "options", options)
    synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt(q_analyses_text, o_analyses_text)
    [raw_synthesized_report] = llm.stage([(synthesizer, prompt_get_synthesized_report, CAP_SYNTH)])
    syn_report = cleansing_syn_report(question, options, raw_synthesized_report)

    all_domains = question_domains + options_domains
    vote_history, revision_history = [], []
    hasno_flag, num_try = True, 0
    while num_try < max_attempt_vote and hasno_flag:
        domain_opinions, revision_advice = {}, {}
        num_try += 1
        hasno_flag = False
        # votes (one stage), then advice from each "no" voter (one stage); same prompts as the sequential loop
        votes = [get_consensus_prompt(domain, syn_report) for domain in all_domains]
        raw = llm.stage([(voter, cons_prompt, CAP_VOTE) for voter, cons_prompt in votes])
        opinions = [cleansing_voting(r) for r in raw]
        no = [i for i, o in enumerate(opinions) if o == "no"]
        advice = llm.stage([(votes[i][0], get_consensus_opinion_prompt(all_domains[i], syn_report), CAP_ADVICE) for i in no])
        adv = dict(zip(no, advice))
        for i, domain in enumerate(all_domains):
            domain_opinions[domain] = opinions[i]
            if opinions[i] == "no":
                revision_advice[domain] = adv[i]
                hasno_flag = True
        if hasno_flag:
            revision_prompt = get_revision_prompt(syn_report, revision_advice)
            [revised_analysis] = llm.stage([("", revision_prompt, CAP_REVISION)])
            syn_report = cleansing_syn_report(question, options, revised_analysis)
            revision_history.append(revision_advice)
        vote_history.append(domain_opinions)

    # final reply (D4)
    [output] = llm.stage([("", get_final_reply_prompt_wsyn(syn_report), None)])
    if output == "ERROR.":
        output = ""
    meta = {"question_domains": question_domains, "options_domains": options_domains, "vote_rounds": num_try,
            "vote_history": vote_history, "n_revisions": len(revision_history)}
    return output, meta


# ---- driver --------------------------------------------------------------------------------------------------------

def sample_pass(rows, store: Store, out: Path, realtime: bool, threads: int) -> int:
    done_path = out / "replies.jsonl"
    done = set()
    if done_path.exists():
        done = {json.loads(x)["prompt_id"] for x in done_path.read_text().splitlines() if x.strip()}
    llm, lock = Opus(store, realtime), threading.Lock()

    def one(row) -> int:
        pid = row["prompt_id"]
        if pid in done:
            return 0
        llm.reset_usage()
        with slot((pid, 0)):
            try:
                text, meta = fully_decode(row["prompt"], llm)
            except Pending:
                return 1
        rec = {"prompt_id": pid, "repeat": 0, "arm": ARM, "text": text, "path": {**meta, "usage": dict(llm.usage())}}
        with lock, done_path.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return 0

    with cf.ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(one, rows))


def grade_pass(rows, store: Store, out: Path, realtime: bool, threads: int) -> int:
    """paper.ablation_staged.grade_pass with this arm only (same graders, same HealthBenchEval call, same format)."""
    import paper.ablation_staged as ab

    ab.ARMS = [ARM]
    return ab.grade_pass(rows, 1, store, out, realtime, threads)


def finalize(out: Path, data: Path) -> None:
    """Add run fields to the grade files, as in the registered arm files."""
    reps = [json.loads(x) for x in (out / "replies.jsonl").read_text().splitlines() if x.strip()]
    tot = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    for r in reps:
        for k in tot:
            tot[k] += r["path"]["usage"][k]
    for key in ("grader", "grader2"):
        p = out / f"{ARM}__{key}.json"
        if p.exists():
            d = json.loads(p.read_text())
            d.update({"subset": "consensus", "backbone": SPEC, "effort": EFFORT, "data": str(data),
                      "usage_per_arm": {"tokens": {SPEC: tot}, "responses_with_usage": len(reps)}})
            p.write_text(json.dumps(d, default=str))


def run(a) -> None:
    store = Store(a.out / "store")
    rows = sorted((json.loads(x) for x in a.data.open()), key=lambda r: r["prompt_id"])
    if a.n:
        rows = rows[: a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "split.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    phases = [lambda: sample_pass(rows, store, a.out, a.realtime, a.threads)]
    if not a.no_grade:
        phases.append(lambda: grade_pass(rows, store, a.out, a.realtime, a.threads))
    for i, phase in enumerate(phases):
        while True:
            while collect(store):
                time.sleep(a.poll)
            missing = phase()
            print(time.strftime("%H:%M:%S"), ["sample", "grade"][i], "missing", missing, "queued", len(store.queue),
                  flush=True)
            if not store.queue:
                if missing and not a.realtime:
                    sys.exit(f"{missing} still missing with nothing to submit; inspect {a.out}")
                break
            jobs = submit(store)
            print("submitted", [(j["vendor"], j["model"], j["n"]) for j in jobs], flush=True)
    finalize(a.out, a.data)
    print(json.dumps(store.cost(), indent=1))


# ---- analysis ------------------------------------------------------------------------------------------------------

def _by_prompt(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for e in json.loads(path.read_text())["example_level"]:
        out.setdefault(e["prompt_id"], []).append(float(e["score"]))
    return out


def _words_registered(path: Path) -> list[int]:
    n = []
    for e in json.loads(path.read_text())["example_level"]:
        c = e["completion"]
        c = c if isinstance(c, list) else ast.literal_eval(c)  # simple-evals stores the message list as its repr
        n.append(len(" ".join(m["content"] for m in c).split()))
    return n


def paired_bootstrap(diffs, n_boot: int = 10_000, seed: int = 20261028) -> dict:
    import numpy as np

    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": int(len(d)), "diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n_boot": n_boot, "seed": seed}


def analyze(out: Path, registered: Path) -> dict:
    import numpy as np

    res: dict = {"arm": ARM, "graders": {}}
    reps = [json.loads(x) for x in (out / "replies.jsonl").read_text().splitlines() if x.strip()]
    for key, suffix in (("grader", ""), ("grader2", "__grader2")):
        med = _by_prompt(out / f"{ARM}__{key}.json")
        g: dict = {"medagents_score": float(np.mean([v[0] for v in med.values()]))}
        for name, label in (("ai_pcp_v3", "compass_minus_medagents"), ("raw_llm", "llm_alone_minus_medagents")):
            reg = _by_prompt(registered / f"{name}{suffix}.json")
            ids = sorted(set(med) & set(reg))
            assert len(ids) == len(med), "every MedAgents conversation must have registered replies"
            g[f"{name}_score_same_conversations"] = float(np.mean([np.mean(reg[i]) for i in ids]))
            g[label] = paired_bootstrap([np.mean(reg[i]) - med[i][0] for i in ids])
        g["grader_spec"] = json.loads((out / f"{ARM}__{key}.json").read_text())["grader"]
        res["graders"][key] = g
    for label in ("compass_minus_medagents", "llm_alone_minus_medagents"):
        # AMD-PCP-002 section 4.5: inconclusive when a 95% CI includes zero or the two graders give opposite signs
        a, b = (res["graders"][k][label] for k in ("grader", "grader2"))
        excludes_zero = all(x["ci95"][0] > 0 or x["ci95"][1] < 0 for x in (a, b))
        res[f"{label}_conclusive"] = bool(excludes_zero and np.sign(a["diff"]) == np.sign(b["diff"]))
    u = np.array([[r["path"]["usage"][k] for k in ("calls", "input_tokens", "output_tokens")] for r in reps], float)
    w = np.array([len(r["text"].split()) for r in reps], float)
    res["per_reply"] = {
        "n_replies": len(reps),
        "calls_mean": float(u[:, 0].mean()), "calls_min": int(u[:, 0].min()), "calls_max": int(u[:, 0].max()),
        "input_tokens_mean": float(u[:, 1].mean()), "output_tokens_mean": float(u[:, 2].mean()),
        "words_mean": float(w.mean()), "words_median": float(np.median(w)),
        "vote_rounds": {str(k): int(sum(r["path"]["vote_rounds"] == k for r in reps)) for k in (1, 2, 3)},
        "empty_replies": int(sum(not r["text"].strip() for r in reps)),
    }
    cost = sum(r["path"]["usage"]["input_tokens"] for r in reps) / 1e6 * 2 + \
        sum(r["path"]["usage"]["output_tokens"] for r in reps) / 1e6 * 10
    res["per_reply"]["sampler_usd_at_batch_price"] = round(cost, 2)
    comp = {}
    for name in ("ai_pcp_v3", "raw_llm"):
        d = json.loads((registered / f"{name}.json").read_text())
        tok = (d.get("usage_per_arm") or {}).get("tokens") or {}
        n = len(d["example_level"])
        comp[name] = {"tokens_per_reply": {s: {k: v / n for k, v in t.items()} for s, t in tok.items()},
                      "words_mean": float(np.mean(_words_registered(registered / f"{name}.json")))}
    res["registered_arms"] = comp
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "analyze"])
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--registered", type=Path)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--headroom", type=int, default=CAPPED_HEADROOM, help="thinking headroom on capped steps")
    a = ap.parse_args()
    globals()["CAPPED_HEADROOM"] = a.headroom
    if a.cmd == "status":
        store = Store(a.out / "store")
        n = sum(1 for _ in (a.out / "replies.jsonl").open()) if (a.out / "replies.jsonl").exists() else 0
        print(json.dumps({"replies": n, "running_batches": collect(store), "cost": store.cost()}, indent=1))
    elif a.cmd == "analyze":
        res = analyze(a.out, a.registered)
        (a.out / "analysis.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    else:
        run(a)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
