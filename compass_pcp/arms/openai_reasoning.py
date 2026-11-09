"""OpenAI reasoning-model client (GPT-5.x / GPT-6.x) for Paper 2 development arms.

compass_pcp.arms.llm.OpenAILLM (frozen with v22) sends max_tokens and temperature, which reasoning models reject. This
client sends max_completion_tokens (output cap plus reasoning headroom) and reasoning_effort instead, with the same
complete() signature and usage accounting, so the frozen comparator chains run unchanged on a GPT-6.1 Sol backbone.
"""

from dataclasses import dataclass, field

from compass_pcp.arms.llm import Usage, _record, _retry


@dataclass
class OpenAIReasoningLLM:
    model: str
    reasoning_effort: str = "high"
    reasoning_headroom: int = 32000
    phi_allowed: bool = False
    vendor: str = "openai"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        import openai

        self._client = openai.OpenAI(timeout=900.0, max_retries=4)

    def complete(self, system, messages, *, max_tokens=800, temperature=0.0):  # temperature unsupported; ignored
        msgs = [{"role": "system", "content": system}, *[m for m in messages if m["role"] in ("user", "assistant")]]
        resp = _retry(lambda: self._client.chat.completions.create(
            model=self.model, messages=msgs, reasoning_effort=self.reasoning_effort,
            max_completion_tokens=max_tokens + self.reasoning_headroom))
        self.usage.calls += 1
        if resp.usage:
            self.usage.input_tokens += resp.usage.prompt_tokens
            self.usage.output_tokens += resp.usage.completion_tokens
        _record(self, resp.usage.prompt_tokens if resp.usage else 0, resp.usage.completion_tokens if resp.usage else 0)
        return resp.choices[0].message.content or ""
