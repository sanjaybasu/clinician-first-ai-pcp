"""compass-pcp: the symbolic layer that turns Compass from a triage bot into an AI PCP.

The LLM in Compass proposes (interviews, parses, drafts). This package decides:
cannot-miss coverage, when a workup can stop, dosing and ordering permission by
lane, monitoring contracts and deferral. Everything here is deterministic and
LLM-free so that every decision can be replayed and audited.
"""

__version__ = "0.1.0"
