"""Illness scripts: one per chief complaint (target about 25).

A script lists candidate diagnoses, each tagged cannot-miss or not, with a
Waymark-population prior and likelihood ratios per feature. Every number must
carry a ``source``; the loader refuses a script with an unsourced number, and
refuses a production load unless a physician has signed off.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Sourced(BaseModel):
    source: str = Field(min_length=3, description="Citation, claims query id, or guideline section")


class LikelihoodRatio(Sourced):
    lr_pos: float = Field(gt=0)
    lr_neg: float = Field(gt=0)
    lr_pos_ci: tuple[float, float] | None = None
    lr_neg_ci: tuple[float, float] | None = None
    group: str | None = Field(default=None, description="Correlated findings share a group; only the strongest applies")

    @model_validator(mode="after")
    def _ci_contains_point(self) -> "LikelihoodRatio":
        for point, ci in ((self.lr_pos, self.lr_pos_ci), (self.lr_neg, self.lr_neg_ci)):
            if ci is not None and not (0 < ci[0] <= point <= ci[1]):
                raise ValueError(f"LR {point} outside its interval {ci}")
        return self


class Prior(Sourced):
    p: float = Field(gt=0, lt=1)


class Thresholds(Sourced):
    """Pauker–Kassirer thresholds for a cannot-miss diagnosis (design doc, step 6).

    Either give ``test`` and ``treat`` directly, or give the decision-analysis
    inputs and let them be computed: B (net benefit of acting when present),
    R (net harm of acting when absent), and the best rule-out test's Se, Sp and
    harm R_tau. With no remote test (Se = 0, Sp = 1, R_tau = 0) the middle zone
    collapses and both thresholds equal R / (B + R).
    """

    test: float | None = Field(default=None, gt=0, lt=1)
    treat: float | None = Field(default=None, gt=0, lt=1)
    benefit: float | None = Field(default=None, gt=0)
    harm: float | None = Field(default=None, gt=0)
    test_se: float = Field(default=0.0, ge=0, le=1)
    test_sp: float = Field(default=1.0, ge=0, le=1)
    test_harm: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _resolve(self) -> "Thresholds":
        if self.test is None or self.treat is None:
            if self.benefit is None or self.harm is None:
                raise ValueError("give test/treat, or benefit and harm")
            t_test, t_treat = pauker_kassirer(self.benefit, self.harm, self.test_se, self.test_sp, self.test_harm)
            self.test, self.treat = t_test, t_treat
        if self.test > self.treat:
            raise ValueError("test threshold must not exceed treat threshold")
        return self


def pauker_kassirer(b: float, r: float, se: float, sp: float, r_tau: float) -> tuple[float, float]:
    """T_test = [(1-Sp)R + R_tau] / [(1-Sp)R + Se B];  T_treat = [Sp R - R_tau] / [Sp R + (1-Se) B]."""
    t_treat = (sp * r - r_tau) / (sp * r + (1 - se) * b)
    denom = (1 - sp) * r + se * b
    t_test = ((1 - sp) * r + r_tau) / denom if denom > 0 else t_treat
    t_test = min(max(t_test, 1e-9), t_treat)
    return t_test, min(max(t_treat, 1e-9), 1 - 1e-9)


class Diagnosis(BaseModel):
    id: str
    label: str
    cannot_miss: bool = False
    miss_weight: float | None = Field(default=None, gt=0, description="u_k; default 1, or 10 for cannot-miss")
    prior: Prior
    thresholds: Thresholds | None = None
    lrs: dict[str, LikelihoodRatio] = Field(default_factory=dict, description="feature_id -> LR")
    red_flags: list[str] = Field(
        default_factory=list,
        description="must_ask features that screen for this diagnosis without a likelihood ratio (they act by red_flag_action)",
    )

    @model_validator(mode="after")
    def _cannot_miss_has_thresholds(self) -> "Diagnosis":
        if self.cannot_miss and self.thresholds is None:
            raise ValueError(f"cannot-miss diagnosis {self.id} needs thresholds")
        return self

    @property
    def u(self) -> float:
        if self.miss_weight is not None:
            return self.miss_weight
        return 10.0 if self.cannot_miss else 1.0


class Feature(BaseModel):
    id: str
    question: str = Field(description="Plain-language prompt; the LLM rewrites it for the patient")
    remote_askable: bool = Field(default=True, description="False for exam, ECG, imaging")
    cost: float = Field(default=0.0, ge=0, description="Dollar cost plus delay term, for C(a)")
    harm: float = Field(default=0.0, ge=0, description="Iatrogenic risk and burden, for H(a)")
    leak: float | None = Field(
        default=None, gt=0, lt=1, description="l_a: P(positive) when no candidate diagnosis is present (noisy-OR leak)"
    )


def sens_spec(lr: LikelihoodRatio) -> tuple[float, float]:
    """Recover sensitivity and specificity from LR+ and LR- (requires LR+ > 1 > LR-)."""
    if not lr.lr_pos > 1 > lr.lr_neg:
        return (0.5, 0.5)
    spec = (lr.lr_pos - 1) / (lr.lr_pos - lr.lr_neg)
    sens = lr.lr_pos * (1 - lr.lr_neg) / (lr.lr_pos - lr.lr_neg)
    return sens, spec


class ClinicalReview(BaseModel):
    approved_by: str | None = None
    approved_on: str | None = None


class IllnessScript(BaseModel):
    complaint: str
    version: str
    lane: str
    features: list[Feature]
    min_age: int = Field(default=18, ge=0, description="Youngest patient the script's sources and lane cover")
    max_age: int | None = None
    must_ask: list[str] = Field(
        default_factory=list,
        description="Red-flag features that must be observed before the encounter can complete, even when priors "
        "alone sit below the test threshold; unobserved after the budget, the case defers",
    )
    red_flag_action: dict[str, Literal["escalate", "defer", "record"]] = Field(
        default_factory=dict,
        description="What a PRESENT must_ask finding does, independent of any likelihood ratio; default escalate. "
        "'record': the feature must be asked before closure, and a present answer acts only through the pathway",
    )
    diagnoses: list[Diagnosis]
    clinical_review: ClinicalReview = Field(default_factory=ClinicalReview)

    @model_validator(mode="after")
    def _lrs_reference_known_features(self) -> "IllnessScript":
        known = {f.id for f in self.features}
        if set(self.red_flag_action) - set(self.must_ask):
            raise ValueError("red_flag_action keys must be must_ask features")
        for d in self.diagnoses:
            if set(d.red_flags) - set(self.must_ask):
                raise ValueError(f"red_flags of {d.id} must be must_ask features")
        if set(self.must_ask) - known:
            raise ValueError(f"must_ask lists undeclared features {sorted(set(self.must_ask) - known)}")
        for d in self.diagnoses:
            unknown = set(d.lrs) - known
            if unknown:
                raise ValueError(f"{d.id} has LRs for undeclared features {sorted(unknown)}")
        return self

    def feature(self, feature_id: str) -> Feature:
        return next(f for f in self.features if f.id == feature_id)

    def leak(self, feature_id: str) -> float:
        """Script value if given, else 1 - mean specificity across diagnoses that use the feature."""
        f = self.feature(feature_id)
        if f.leak is not None:
            return f.leak
        specs = [sens_spec(d.lrs[feature_id])[1] for d in self.diagnoses if feature_id in d.lrs]
        return min(max(1 - sum(specs) / len(specs), 0.01), 0.99) if specs else 0.5

    @property
    def cannot_miss(self) -> list[Diagnosis]:
        return [d for d in self.diagnoses if d.cannot_miss]


def load_script(path: Path, *, require_review: bool = True) -> IllnessScript:
    script = IllnessScript.model_validate(yaml.safe_load(path.read_text()))
    if require_review and not script.clinical_review.approved_by:
        raise ValueError(f"{path.name} has no physician sign-off; load with require_review=False only in tests")
    return script
