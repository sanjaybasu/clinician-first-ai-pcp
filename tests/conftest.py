from pathlib import Path

import pytest

from compass_pcp.engine.scripts import IllnessScript, load_script
from compass_pcp.findings import CodeSystem, Finding, Polarity, Provenance

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def script() -> IllnessScript:
    return load_script(FIXTURES / "synthetic_script.yaml")


def make_finding(
    feature_id: str,
    polarity: Polarity = Polarity.PRESENT,
    provenance: Provenance = Provenance.PATIENT_REPORT,
    extractor: str = "ext-a",
    value: float | None = None,
) -> Finding:
    return Finding(
        feature_id=feature_id,
        code_system=CodeSystem.LOCAL,
        code=feature_id,
        polarity=polarity,
        provenance=provenance,
        extractor_id=extractor,
        value=value,
    )
