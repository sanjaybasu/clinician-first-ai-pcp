"""COMPASS clinician-facing pipeline (Paper 2 development): two independent drafts, a review, one final writer.

The physician-facing half of the AI PCP answers a clinician's request (a consult question, a note or letter, an evidence
question). The model drafts; the structure around it targets the two failure modes seen in the comparator chains on the
HealthBench Professional development split: omitted content and unsupported or fabricated content.

1. Draft A: Opus 5.5 (adaptive thinking) with web search, so guideline and trial statements can be checked against
   current sources.
2. Draft B: GPT-6.1 Sol (high reasoning), a different developer, written independently.
3. Review: GPT-6.1 Sol reads the request and both drafts and lists (a) what the request asks for, in what format and
   length; (b) statements in either draft that are wrong, unsupported, out of date or not given by the user; (c) content
   an expert specialist would expect that both drafts omit; (d) points where the drafts disagree, with the better-supported
   side.
4. Final: Opus 5.5 writes the one answer from the request, both drafts and the review, with a length target.
5. Edit (edit=True): Opus 5.5 shortens the answer to about 2,000 characters without adding facts, because the
   benchmark subtracts 2.94e-5 per character above 2,000 and physicians read short answers faster.

Flags give the ablations: search=False drops web search; dual=False uses Draft A only; review=False merges without the
review step; second="opus" makes Draft B and the review Opus 5.5 calls (single-developer pipeline); final=False skips the
final writer, so the editor works on Draft A (with dual=False and review=False: one searched draft plus the editor). Benchmark tasks hold no PHI; on Waymark data only BAA vendors are called (Anthropic, OpenAI), and web
search queries must carry no identifiers.
"""

from dataclasses import dataclass, field

from compass_pcp.arms.llm import Usage
from compass_pcp.arms.openai_reasoning import OpenAIReasoningLLM

DRAFT_SYS = (
    "You support physicians and other health professionals. Do exactly the task asked, for the stated reader, in the "
    "format asked. Be accurate and specific: name the relevant guideline or trial with its year when you rely on one, "
    "state how strong the evidence is, give red flags, contraindications and the practical next step. Do not invent "
    "patient details, references, trial results or numbers; if something needed is missing, say what and why it "
    "matters, then answer under stated assumptions. Use the user's language."
)
SEARCH_NOTE = (
    " Use web search to check any guideline, trial, drug-safety or recommendation statement that may have changed in "
    "the last three years, and any reference you cite."
)
REVIEW_SYS = (
    "You are a senior specialist reviewing two draft answers to a health professional's request before one final "
    "answer is written. Do not write the answer. Write terse review notes under five headings:\n"
    "REQUEST: what is asked, for which reader, the required format, and any facts the user gave that the answer must "
    "keep exactly.\n"
    "USER-DATA ERRORS: anything in the user's own information or plan that is wrong or unsafe (a drug, dose, device "
    "size, fluid, test or interpretation) and that the answer must name directly with the correct alternative.\n"
    "ERRORS: each statement in either draft that is wrong, out of date, unsupported, overstated, fabricated (patient "
    "details, references, numbers not given or not verifiable), unsafe, or that departs from what the user provided. "
    "Say which draft and why.\n"
    "OMISSIONS: content an expert in this specialty would expect and both drafts lack (key differentials, red flags, "
    "contraindications and interactions, evidence strength and its limits, current guideline positions, practical "
    "next steps, patient-specific considerations from the request). Only items that matter for this request.\n"
    "CONFLICTS: where the drafts disagree, which side is better supported and why."
)
FINAL_SYS = (
    "Write the single final answer to the health professional's request. You have two independent drafts and a "
    "specialist's review. Keep what is correct in either draft, fix every error the review lists, add the omissions "
    "that matter, settle conflicts as the review advises unless you have a clear reason not to, and never add "
    "unverified references, numbers or patient details. State each user-data error plainly with the correct "
    "alternative; do not hedge it. Follow the requested format and reader exactly. Be concise: at most about 2,500 "
    "characters, unless the user explicitly asked for a long document (a full note, letter or handout); cut lower-value "
    "detail before cutting a needed point. Use light formatting. No preamble about the drafts or the review; output only "
    "the answer."
)

EDIT_SYS = (
    "You edit a clinical answer for a busy physician. Shorten it to at most 2,000 characters unless the user explicitly "
    "asked for a long document (a full note, letter or handout), in which case keep the document and only remove "
    "repetition. Keep every clinically important point: the direct answer, corrections of errors in the user's own "
    "information, red flags, contraindications, evidence strength and the practical next step. Remove repetition, "
    "generic background, hedging boilerplate and headings that add no content. Do not add new facts. Output only the "
    "edited answer."
)


def _transcript(conv: list[dict]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in conv)


@dataclass
class ProPipeline:
    search: bool = True
    dual: bool = True
    review: bool = True
    edit: bool = False
    second: str = "gpt"
    final: bool = True
    usage: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        import anthropic

        self._a = anthropic.Anthropic(base_url="https://api.anthropic.com", timeout=900.0, max_retries=4)
        self._g = OpenAIReasoningLLM("gpt-6.1-sol")

    def _add(self, who: str, i: int, o: int) -> None:
        u = self.usage.setdefault(who, [0, 0])
        u[0] += i
        u[1] += o

    def _opus(self, system: str, messages: list[dict], search: bool, max_tokens: int = 16000) -> str:
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}] if search else []
        m = None
        for _ in range(4):
            kw = {"tools": tools} if tools else {}
            with self._a.messages.stream(model="claude-opus-5-5", max_tokens=max_tokens, thinking={"type": "adaptive"},
                                         system=system, messages=msgs, **kw) as s:
                m = s.get_final_message()
            self._add("opus55", m.usage.input_tokens, m.usage.output_tokens)
            st = getattr(m.usage, "server_tool_use", None)
            if st is not None:
                self._add("web_search", getattr(st, "web_search_requests", 0) or 0, 0)
            if m.stop_reason != "pause_turn":
                break
            msgs = msgs + [{"role": "assistant", "content": m.content}]
        return "".join(b.text for b in m.content if b.type == "text").strip()

    def _gpt(self, system: str, messages: list[dict], max_tokens: int = 6000) -> str:
        before = (self._g.usage.input_tokens, self._g.usage.output_tokens)
        out = self._g.complete(system, messages, max_tokens=max_tokens)
        self._add("gpt61sol", self._g.usage.input_tokens - before[0], self._g.usage.output_tokens - before[1])
        return out.strip()

    def edit_only(self, conv: list[dict], answer: str) -> str:
        """Step 5 (length editor): Opus 5.5 shortens the final answer; reused on stored step-4 answers in development."""
        msg = f"REQUEST (conversation so far):\n{_transcript(conv)}\n\nANSWER TO EDIT:\n{answer}"
        return self._opus(EDIT_SYS, [{"role": "user", "content": msg}], search=False, max_tokens=8000)

    def answer(self, conv: list[dict]) -> tuple[str, dict]:
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(2) as ex:
            fa = ex.submit(self._opus, DRAFT_SYS + (SEARCH_NOTE if self.search else ""), conv, self.search)
            fb = (ex.submit(self._gpt, DRAFT_SYS, conv) if self.second == "gpt" else
                  ex.submit(self._opus, DRAFT_SYS, conv, False)) if self.dual else None
            a, b = fa.result(), (fb.result() if fb else "")
        drafts = f"DRAFT 1:\n{a}" + (f"\n\nDRAFT 2:\n{b}" if self.dual else "")
        req = f"REQUEST (conversation so far):\n{_transcript(conv)}\n\n{drafts}"
        rmsg = [{"role": "user", "content": req}]
        notes = ((self._gpt(REVIEW_SYS, rmsg, max_tokens=4000) if self.second == "gpt" else
                  self._opus(REVIEW_SYS, rmsg, search=False, max_tokens=8000)) if self.review else "")
        final_in = req + (f"\n\nREVIEW NOTES:\n{notes}" if notes else "")
        final = (self._opus(FINAL_SYS, [{"role": "user", "content": final_in}], search=False, max_tokens=12000)
                 if self.final else a)
        if self.edit:
            final = self.edit_only(conv, final)
        return final, {"draft_a_chars": len(a), "draft_b_chars": len(b), "review_chars": len(notes)}
