from datetime import date
from pathlib import Path

from compass_pcp.engine.truth import FALSE, TRUE, UNKNOWN, Bounds
from compass_pcp.gate import FindingKind, Owner, evaluate_lab_recency, evaluate_rules, lab_bounds, load_rules

RULES_DIR = Path(__file__).resolve().parents[1] / "compass_pcp" / "data" / "rules"
TITRATION = load_rules(RULES_DIR / "titration.yaml")
CONTRACEPTION = load_rules(RULES_DIR / "contraception.yaml")
REFILL = load_rules(RULES_DIR / "refill.yaml")
TODAY = date(2026, 10, 27)
FRESH = date(2026, 10, 20)


def _names(term: str | list[str]) -> list[str]:
    return [term] if isinstance(term, str) else term


def _closed(rules, prefix: str) -> list[str]:
    return sorted({n for r in rules for t in r.when_all for n in _names(t) if n.startswith(prefix)})


TITRATE = _closed(TITRATION.rules, "titrate:")
METHODS = _closed(CONTRACEPTION.rules, "rx_class:")


def titrate(*classes: str, **facts: Bounds) -> dict[str, Bounds]:
    """Closed world over titrate:<class>: requested classes TRUE, every other FALSE."""
    preds = {c: FALSE for c in TITRATE}
    preds.update({f"titrate:{c}": TRUE for c in classes})
    preds.update(facts)
    return preds


def renew(*methods: str, **facts: Bounds) -> dict[str, Bounds]:
    """Closed world over the contraceptive rx_class values."""
    preds = {c: FALSE for c in METHODS}
    preds.update({f"rx_class:{m}": TRUE for m in methods})
    preds.update(facts)
    return preds


def lab(value: float | None, on: date | None, op: str, threshold: float, max_age_days: int = 28) -> Bounds:
    return lab_bounds(value, on, op=op, threshold=threshold, max_age_days=max_age_days, today=TODAY)


def ids(findings) -> set[str]:
    return {f.rule_id for f in findings}


def by_id(findings):
    return {f.rule_id: f for f in findings}


def test_files_load_and_every_rule_is_sourced():
    for rs in (TITRATION, CONTRACEPTION):
        assert rs.version == "1.0.0" and rs.clinical_review.physician_approved_by
        assert all(len(r.source) > 10 for r in rs.rules + rs.lab_recency)
    all_ids = [r.id for rs in (REFILL, TITRATION, CONTRACEPTION) for r in rs.rules + rs.lab_recency]
    assert len(all_ids) == len(set(all_ids))


# ---- Blood pressure titration ----

HTN_OK = {
    "bp:at_or_below_individual_goal": FALSE,
    "bp:fewer_than_2_readings_on_2_occasions": FALSE,
    "sx:symptomatic_hypotension": FALSE,
    "preg:possible": FALSE,
    "med:ace_inhibitor": FALSE,
    "med:arb": FALSE,
    "med:arni": FALSE,
    "med:direct_renin_inhibitor": FALSE,
    "delta:creatinine_rise_gt_30pct_after_last_rasi_change": FALSE,
    "monitor:no_bmp_2_to_4_weeks_after_last_rasi_change": FALSE,
}


def test_ace_uptitration_passes_fires_and_holds_on_potassium():
    def run(k, on=FRESH):
        return evaluate_rules(TITRATION.rules, titrate(
            "antihypertensive", "ace_inhibitor", **HTN_OK, **{"lab:potassium_gt_5_5": lab(k, on, "gt", 5.5)}))
    assert run(4.6) == []
    fired = by_id(run(5.8))
    assert set(fired) == {"rasi_titration_potassium_gt_5_5"}
    assert fired["rasi_titration_potassium_gt_5_5"].owner == Owner.PHARMACIST
    stale = by_id(run(4.6, date(2026, 8, 1)))
    assert stale["rasi_titration_potassium_gt_5_5"].unresolved_predicates == ["lab:potassium_gt_5_5"]


def test_bp_at_goal_blocks_and_unknown_goal_holds():
    facts = dict(HTN_OK, **{"lab:potassium_gt_5_5": FALSE})
    at_goal = ids(evaluate_rules(TITRATION.rules, titrate("antihypertensive", "ace_inhibitor", **dict(facts, **{"bp:at_or_below_individual_goal": TRUE}))))
    assert at_goal == {"htn_bp_at_or_below_goal"}
    no_goal = by_id(evaluate_rules(TITRATION.rules, titrate("antihypertensive", "ace_inhibitor", **dict(facts, **{"bp:at_or_below_individual_goal": UNKNOWN}))))
    assert no_goal["htn_bp_at_or_below_goal"].unresolved_predicates == ["bp:at_or_below_individual_goal"]


def test_symptomatic_hypotension_is_out_of_lane():
    f = by_id(evaluate_rules(TITRATION.rules, titrate("antihypertensive", "thiazide", **dict(
        HTN_OK, **{"sx:symptomatic_hypotension": TRUE, "ref:potassium_below_lln": FALSE}))))
    assert set(f) == {"htn_symptomatic_hypotension"}
    assert f["htn_symptomatic_hypotension"].kind == FindingKind.OUT_OF_LANE
    assert f["htn_symptomatic_hypotension"].owner == Owner.ORDERING_CLINICIAN


def test_dual_ras_blockade_blocks():
    f = ids(evaluate_rules(TITRATION.rules, titrate("antihypertensive", "arb", **dict(
        HTN_OK, **{"med:ace_inhibitor": TRUE, "lab:potassium_gt_5_5": FALSE}))))
    assert f == {"rasi_dual_blockade"}


def test_mra_uptitration_potassium_4_8_boundary():
    def run(k):
        return ids(evaluate_rules(TITRATION.rules, titrate("antihypertensive", "mra", **dict(
            HTN_OK, **{"lab:potassium_gt_4_8": lab(k, FRESH, "gt", 4.8)}))))
    assert run(4.8) == set()
    assert run(4.9) == {"mra_titration_potassium_gt_4_8"}


# ---- Glucose titration ----

DM_OK = {
    "glycemia:a1c_at_or_below_individual_goal": FALSE,
    "hx:hypoglycemia_level_2_or_3_since_last_change": FALSE,
    "hx:glucose_lt_70_since_last_change": FALSE,
    "smbg:fasting_lt_80_since_last_change": FALSE,
    "glycemia:overbasalization_signs": FALSE,
    "dx:high_hypoglycemia_risk": FALSE,
}


def test_basal_insulin_uptitration_hypoglycemia():
    assert evaluate_rules(TITRATION.rules, titrate("glucose_lowering", "basal_insulin", **DM_OK)) == []
    lvl1 = ids(evaluate_rules(TITRATION.rules, titrate("glucose_lowering", "basal_insulin", **dict(DM_OK, **{"hx:glucose_lt_70_since_last_change": TRUE}))))
    assert lvl1 == {"hypoglycemic_agent_titration_glucose_lt_70"}
    lvl2 = by_id(evaluate_rules(TITRATION.rules, titrate("glucose_lowering", "basal_insulin", **dict(DM_OK, **{"hx:hypoglycemia_level_2_or_3_since_last_change": TRUE}))))
    assert lvl2["dm_titration_level_2_or_3_hypoglycemia"].owner == Owner.ORDERING_CLINICIAN
    asked = by_id(evaluate_rules(TITRATION.rules, titrate("glucose_lowering", "basal_insulin", **dict(DM_OK, **{"smbg:fasting_lt_80_since_last_change": UNKNOWN}))))
    assert asked["basal_insulin_titration_fasting_lt_80"].unresolved_predicates == ["smbg:fasting_lt_80_since_last_change"]


def test_semaglutide_escalation_interval():
    def run(lt_28):
        return ids(evaluate_rules(TITRATION.rules, titrate(
            "glucose_lowering", "glp1_ra", "semaglutide", **dict(DM_OK, **{"titration:days_on_current_dose_lt_28": lt_28}))))
    assert run(TRUE) == {"weekly_glp1_titration_lt_4_weeks"}
    assert run(FALSE) == set()


def test_a1c_recency_before_glucose_titration():
    assert evaluate_lab_recency(TITRATION.lab_recency, {"glucose_lowering"}, {"17856-6": date(2026, 9, 1)}, TODAY) == []
    old = evaluate_lab_recency(TITRATION.lab_recency, {"glucose_lowering"}, {"4548-4": date(2026, 6, 1)}, TODAY)
    assert [f.rule_id for f in old] == ["a1c_before_glucose_titration"]
    assert evaluate_lab_recency(TITRATION.lab_recency, {"antihypertensive"}, {}, TODAY) == []


# ---- Contraception renewal ----

CHC_OK = {
    "age_ge_35": FALSE,
    "tob:smokes_lt_15_per_day": FALSE,
    "tob:smokes_ge_15_per_day": FALSE,
    "bp:sbp_ge_140": FALSE,
    "bp:dbp_ge_90": FALSE,
    "bp:sbp_ge_160": FALSE,
    "bp:dbp_ge_100": FALSE,
    "postpartum:lt_21_days": FALSE,
    "postpartum:21_to_42_days": FALSE,
    "postpartum:21_to_lt_30_days": FALSE,
}
CHC_FALSE_PREFIXES = ("dx:", "hx:", "med:")
CHC_NAMES = {n for r in CONTRACEPTION.rules for t in r.when_all for n in _names(t)}


def chc_clean(**overrides: Bounds) -> dict[str, Bounds]:
    facts = {n: FALSE for n in CHC_NAMES if n.startswith(CHC_FALSE_PREFIXES)}
    facts.update(CHC_OK)
    facts.update({k: v for k, v in overrides.items()})
    return facts


def test_chc_clean_record_passes():
    assert evaluate_rules(CONTRACEPTION.rules, renew("chc", "coc", **chc_clean())) == []


def test_chc_smoking_age_35_category_3_and_4():
    cat3 = by_id(evaluate_rules(CONTRACEPTION.rules, renew("chc", **chc_clean(age_ge_35=TRUE, **{"tob:smokes_lt_15_per_day": TRUE}))))
    assert set(cat3) == {"chc_age_ge_35_smokes_lt_15"}
    assert cat3["chc_age_ge_35_smokes_lt_15"].owner == Owner.PHARMACIST
    cat4 = by_id(evaluate_rules(CONTRACEPTION.rules, renew("chc", **chc_clean(age_ge_35=TRUE, **{"tob:smokes_ge_15_per_day": TRUE}))))
    assert set(cat4) == {"chc_age_ge_35_smokes_ge_15"}
    assert cat4["chc_age_ge_35_smokes_ge_15"].kind == FindingKind.OUT_OF_LANE
    young = evaluate_rules(CONTRACEPTION.rules, renew("chc", **chc_clean(**{"tob:smokes_ge_15_per_day": TRUE})))
    assert young == []


def test_chc_migraine_with_aura_unknown_asks():
    facts = chc_clean()
    del facts["dx:migraine_with_aura"]
    held = by_id(evaluate_rules(CONTRACEPTION.rules, renew("chc", **facts)))
    assert set(held) == {"chc_migraine_with_aura"}
    assert held["chc_migraine_with_aura"].unresolved_predicates == ["dx:migraine_with_aura"]


def test_chc_blood_pressure_tiers():
    cat3 = ids(evaluate_rules(CONTRACEPTION.rules, renew("chc", **chc_clean(**{"bp:sbp_ge_140": TRUE}))))
    assert cat3 == {"chc_hypertension"}
    cat4 = ids(evaluate_rules(CONTRACEPTION.rules, renew("chc", **chc_clean(**{"bp:sbp_ge_140": TRUE, "bp:sbp_ge_160": TRUE}))))
    assert cat4 == {"chc_hypertension", "chc_bp_ge_160_100_or_vascular_disease"}


def test_pop_is_not_held_by_chc_only_conditions():
    facts = chc_clean(age_ge_35=TRUE, **{"tob:smokes_ge_15_per_day": TRUE, "dx:migraine_with_aura": TRUE})
    assert evaluate_rules(CONTRACEPTION.rules, renew("pop", **facts)) == []
    f = ids(evaluate_rules(CONTRACEPTION.rules, renew("pop", **chc_clean(**{"med:rifampin_or_rifabutin": TRUE}))))
    assert f == {"pop_interacting_drug"}


def test_coc_bariatric_applies_to_pill_not_ring():
    facts = chc_clean(**{"hx:malabsorptive_bariatric_surgery": TRUE})
    assert ids(evaluate_rules(CONTRACEPTION.rules, renew("chc", "coc", **facts))) == {"coc_malabsorptive_bariatric_surgery"}
    assert evaluate_rules(CONTRACEPTION.rules, renew("chc", **facts)) == []
