"""SDBench-style gatekeeper over a real episode.

The architecture sees only the opening message and that day's Clinical Context.
The gatekeeper holds the full record and answers only what is asked. It never
invents a symptom, exam finding or result: an absent fact returns
NOT_DOCUMENTED, and an ordered test with no real result in the plausible window
returns NOT_AVAILABLE. Every request and answer is logged for the leakage and
fabrication audit (two physicians, 100 episodes, both rates under 2%).

Paraphrase into the patient's register is optional and pluggable. The
paraphraser receives only the single answered fact, never the record, and must
come from a different vendor than the system under test.
"""

from datetime import timedelta
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from compass_pcp.findings import Polarity
from compass_pcp.harness.episode import Episode


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NOT_DOCUMENTED = "not_documented"
    NOT_AVAILABLE = "not_available"


class GatekeeperAnswer(BaseModel):
    request: str
    status: AnswerStatus
    polarity: Polarity | None = None
    value: float | None = None
    text: str


class Paraphraser(Protocol):
    vendor: str

    def paraphrase(self, fact_text: str, patient_style_examples: list[str]) -> str: ...


class Gatekeeper:
    def __init__(self, episode: Episode, paraphraser: Paraphraser | None = None) -> None:
        self._episode = episode
        self._paraphraser = paraphraser
        self.log: list[GatekeeperAnswer] = []

    def ask(self, feature_id: str, question_text: str) -> GatekeeperAnswer:
        fact = self._episode.record_facts.get(feature_id)
        if fact is None:
            answer = GatekeeperAnswer(request=feature_id, status=AnswerStatus.NOT_DOCUMENTED, text="not documented")
        else:
            text = fact.patient_phrasing or self._template(question_text, fact.polarity)
            if self._paraphraser is not None and fact.patient_phrasing is None:
                text = self._paraphraser.paraphrase(text, [self._episode.opening_message])
            answer = GatekeeperAnswer(
                request=feature_id, status=AnswerStatus.ANSWERED, polarity=fact.polarity, value=fact.value, text=text
            )
        self.log.append(answer)
        return answer

    def order(self, loinc: str, window_days: int) -> GatekeeperAnswer:
        start = self._episode.index_time
        end = start + timedelta(days=window_days)
        hits = [r for r in self._episode.labs if r.loinc == loinc and start <= r.resulted_at <= end]
        if not hits:
            answer = GatekeeperAnswer(request=f"lab:{loinc}", status=AnswerStatus.NOT_AVAILABLE, text="not available")
        else:
            first = min(hits, key=lambda r: r.resulted_at)
            answer = GatekeeperAnswer(
                request=f"lab:{loinc}", status=AnswerStatus.ANSWERED, value=first.value, text=f"{first.value} {first.unit or ''}".strip()
            )
        self.log.append(answer)
        return answer

    @staticmethod
    def _template(question_text: str, polarity: Polarity) -> str:
        return ("yes" if polarity == Polarity.PRESENT else "no") + f" ({question_text})"
