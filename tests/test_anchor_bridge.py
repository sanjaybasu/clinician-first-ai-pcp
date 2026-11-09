import pytest

from compass_pcp import anchor_bridge
from compass_pcp.gate import FindingKind

try:
    anchor_bridge.anchor_repo()
    HAVE_ANCHOR = True
except FileNotFoundError:
    HAVE_ANCHOR = False

pytestmark = pytest.mark.skipif(not HAVE_ANCHOR, reason="ANCHOR_REPO not available")


def test_corpus_loads_full_size():
    assert len(anchor_bridge.light_anchor().rules) > 3000


def test_plan_omission_becomes_blocking_finding():
    f = anchor_bridge.plan_findings("on warfarin, new seizure today. Plan: rest at home.")
    assert f and all(x.kind in {FindingKind.CANNOT_MISS_OMISSION, FindingKind.COMMISSION} for x in f)
    assert anchor_bridge.anchor_commit() != "unknown"
