from datetime import date
from pathlib import Path

import yaml

from compass_pcp.engine.truth import FALSE, TRUE, UNKNOWN, Bounds
from compass_pcp.gate import FindingKind, Owner, evaluate_lab_recency, evaluate_rules, lab_bounds, load_rules
from compass_pcp.monitoring import ContractStatus, LabValue, MonitoringContract, evaluate_contract

RULES_DIR = Path(__file__).resolve().parents[1] / "compass_pcp" / "data" / "rules"
RULESET = load_rules(RULES_DIR / "refill.yaml")
CONTRACTS = [MonitoringContract.model_validate(c) for c in yaml.safe_load((RULES_DIR / "refill_monitoring.yaml").read_text())["contracts"]]
TODAY = date(2026, 10, 27)
FRESH = date(2026, 10, 1)
STALE = date(2025, 1, 1)


def _names(term: str | list[str]) -> list[str]:
    return [term] if isinstance(term, str) else term


RX_CLASSES = sorted({n for r in RULESET.rules for t in r.when_all for n in _names(t) if n.startswith("rx_class:")})


def refill(rx: str, **facts: Bounds) -> dict[str, Bounds]:
    """Closed world over rx_class: the requested class is TRUE, every other class is FALSE.

    controlled_substance defaults to FALSE because that rule has no rx_class term.
    """
    preds = {c: FALSE for c in RX_CLASSES}
    preds["controlled_substance"] = FALSE
    preds[f"rx_class:{rx}"] = TRUE
    preds.update({k.replace("__", ":"): v for k, v in facts.items()})
    return preds


def lab(value: float | None, on: date | None, op: str, threshold: float, max_age_days: int = 365) -> Bounds:
    return lab_bounds(value, on, op=op, threshold=threshold, max_age_days=max_age_days, today=TODAY)


def by_id(findings):
    return {f.rule_id: f for f in findings}


def test_files_load_and_every_rule_is_sourced():
    assert RULESET.version == "1.0.0" and RULESET.clinical_review.physician_approved_by
    ids = [r.id for r in RULESET.rules] + [r.id for r in RULESET.lab_recency] + [c.id for c in CONTRACTS]
    assert len(ids) == len(set(ids))
    assert all(len(r.source) > 10 for r in RULESET.rules + RULESET.lab_recency + CONTRACTS)


def test_ace_inhibitor_fires_holds_and_passes():
    base = {"preg:possible": FALSE, "allergy_class:ace_inhibitor": FALSE, "lab:egfr_lt_15": lab(60, FRESH, "lt", 15)}
    high = by_id(evaluate_rules(RULESET.rules, refill("ace_inhibitor", **{"lab:potassium_gt_5_5": lab(5.8, FRESH, "gt", 5.5)}, **base)))
    assert high["rasi_potassium_gt_5_5"].unresolved_predicates == []
    assert high["rasi_potassium_gt_5_5"].owner == Owner.PHARMACIST
    ok = evaluate_rules(RULESET.rules, refill("ace_inhibitor", **{"lab:potassium_gt_5_5": lab(4.2, FRESH, "gt", 5.5)}, **base))
    assert ok == []
    stale = by_id(evaluate_rules(RULESET.rules, refill("ace_inhibitor", **{"lab:potassium_gt_5_5": lab(4.2, STALE, "gt", 5.5)}, **base)))
    assert stale["rasi_potassium_gt_5_5"].unresolved_predicates == ["lab:potassium_gt_5_5"]


def test_arb_pregnancy_unknown_holds():
    preds = refill("arb", **{"lab:potassium_gt_5_5": FALSE, "lab:egfr_lt_15": FALSE, "allergy_class:arb": FALSE})
    held = by_id(evaluate_rules(RULESET.rules, preds))
    assert set(held) == {"rasi_pregnancy_possible"}
    assert held["rasi_pregnancy_possible"].unresolved_predicates == ["preg:possible"]


def test_metformin_egfr_thresholds():
    def run(egfr):
        return set(by_id(evaluate_rules(RULESET.rules, refill(
            "metformin", **{"lab:egfr_lt_30": lab(egfr, FRESH, "lt", 30), "lab:egfr_lt_45": lab(egfr, FRESH, "lt", 45)}))))
    assert run(28) == {"metformin_egfr_lt_30", "metformin_egfr_lt_45"}
    assert run(40) == {"metformin_egfr_lt_45"}
    assert run(60) == set()


def test_apixaban_two_of_three():
    def run(age80, wt60, cr):
        preds = refill("apixaban", age_ge_80=age80, weight_le_60_kg=wt60, **{"lab:creatinine_ge_1_5": cr})
        return by_id(evaluate_rules(RULESET.rules, preds)).get("apixaban_dose_reduction_criteria")
    assert run(TRUE, TRUE, lab(1.0, FRESH, "ge", 1.5)).unresolved_predicates == []
    assert run(TRUE, FALSE, lab(1.0, FRESH, "ge", 1.5)) is None
    assert run(FALSE, TRUE, lab(1.8, FRESH, "ge", 1.5)).unresolved_predicates == []
    held = run(TRUE, FALSE, lab(None, None, "ge", 1.5))
    assert held is not None and "lab:creatinine_ge_1_5" in held.unresolved_predicates
    # Label says ">= 1.5 mg/dL": the boundary value fires.
    assert run(FALSE, TRUE, lab(1.5, FRESH, "ge", 1.5)).unresolved_predicates == []


def test_always_out_of_lane():
    stim = by_id(evaluate_rules(RULESET.rules, refill("statin", controlled_substance=TRUE, **{"preg:possible": FALSE, "allergy_class:statin": FALSE})))
    assert set(stim) == {"controlled_substance_out_of_lane"}
    warf = by_id(evaluate_rules(RULESET.rules, refill("warfarin", controlled_substance=FALSE)))
    assert set(warf) == {"warfarin_route_pharmacist"}
    iso = by_id(evaluate_rules(RULESET.rules, refill("isotretinoin", controlled_substance=FALSE)))
    assert set(iso) == {"isotretinoin_out_of_lane"}


def test_allergy_cross_class_pattern():
    preds = refill("thiazide", controlled_substance=FALSE, **{"allergy_class:sulfonamide": TRUE})
    f = by_id(evaluate_rules(RULESET.rules, preds))
    assert f["thiazide_sulfonamide_allergy"].kind == FindingKind.ALLERGY


def test_non_requested_class_passes():
    preds = refill("levothyroxine", controlled_substance=FALSE)
    assert evaluate_rules(RULESET.rules, preds) == []


def test_lithium_recency():
    fresh = {"14334-7": date(2026, 9, 15), "3016-3": date(2026, 6, 1), "2160-0": date(2026, 6, 1)}
    assert evaluate_lab_recency(RULESET.lab_recency, {"lithium"}, fresh, TODAY) == []
    old_tsh = dict(fresh, **{"3016-3": date(2026, 3, 1)})
    assert [f.rule_id for f in evaluate_lab_recency(RULESET.lab_recency, {"lithium"}, old_tsh, TODAY)] == ["lithium_tsh"]
    assert {f.rule_id for f in evaluate_lab_recency(RULESET.lab_recency, {"lithium"}, {}, TODAY)} == {"lithium_level", "lithium_tsh", "lithium_egfr"}


def test_ace_monitoring_contract_breach_on_creatinine():
    c = next(c for c in CONTRACTS if c.id == "ace_inhibitor_start_or_dose_change")
    start, base = date(2026, 10, 1), {"2160-0": 1.0, "2823-3": 4.3}
    ok = [LabValue(loinc="2160-0", value=1.2, on=date(2026, 10, 20)), LabValue(loinc="2823-3", value=4.6, on=date(2026, 10, 20))]
    assert evaluate_contract(c, start, base, ok, TODAY) == ContractStatus.OK
    rise = [LabValue(loinc="2160-0", value=1.35, on=date(2026, 10, 20)), ok[1]]
    assert evaluate_contract(c, start, base, rise, TODAY) == ContractStatus.BREACH
    assert evaluate_contract(c, start, base, [], date(2026, 11, 5)) == ContractStatus.OVERDUE


def test_rivaroxaban_crcl_boundary_le_50():
    def run(crcl):
        preds = refill("rivaroxaban", **{"lab:crcl_lt_15": lab(crcl, FRESH, "lt", 15), "lab:crcl_le_50": lab(crcl, FRESH, "le", 50)})
        return set(by_id(evaluate_rules(RULESET.rules, preds)))
    assert run(50) == {"rivaroxaban_crcl_le_50"}
    assert run(51) == set()
    assert run(10) == {"rivaroxaban_crcl_lt_15", "rivaroxaban_crcl_le_50"}


def test_lithium_toxic_level_routes_to_ordering_clinician():
    preds = refill("lithium", **{"lab:lithium_gt_1_2": lab(1.5, FRESH, "gt", 1.2, 60), "lab:lithium_ge_1_5": lab(1.5, FRESH, "ge", 1.5, 60)})
    f = by_id(evaluate_rules(RULESET.rules, preds))
    assert set(f) == {"lithium_level_gt_1_2", "lithium_level_ge_1_5"}
    assert f["lithium_level_ge_1_5"].owner == Owner.ORDERING_CLINICIAN


def test_out_of_lane_kinds_route_to_ordering_clinician():
    pregabalin = by_id(evaluate_rules(RULESET.rules, refill("pregabalin", controlled_substance=TRUE, **{"med:opioid": FALSE})))
    assert set(pregabalin) == {"controlled_substance_out_of_lane"}
    assert pregabalin["controlled_substance_out_of_lane"].kind == FindingKind.OUT_OF_LANE
    assert pregabalin["controlled_substance_out_of_lane"].owner == Owner.ORDERING_CLINICIAN


def test_beta_blocker_conduction_and_asthma():
    base = {"dx:av_block_gt_first_degree_or_sick_sinus_without_pacemaker": FALSE}
    carvedilol = lambda asthma: by_id(evaluate_rules(RULESET.rules, refill(
        "carvedilol", **{"rx_class:beta_blocker": TRUE, "dx:asthma": asthma}, **base)))
    assert set(carvedilol(TRUE)) == {"nonselective_beta_blocker_asthma"}
    assert carvedilol(FALSE) == {}
    held = carvedilol(UNKNOWN)
    assert held["nonselective_beta_blocker_asthma"].unresolved_predicates == ["dx:asthma"]
    metoprolol = evaluate_rules(RULESET.rules, refill("beta_blocker", **{"dx:av_block_gt_first_degree_or_sick_sinus_without_pacemaker": TRUE}))
    assert [f.rule_id for f in metoprolol] == ["av_nodal_agent_conduction_disease"]


def test_nondhp_ccb_sbp_lt_90_uses_vital():
    def run(sbp_lt_90):
        return set(by_id(evaluate_rules(RULESET.rules, refill(
            "nondhp_ccb", **{"vital:sbp_lt_90": sbp_lt_90, "dx:hfref": FALSE,
                             "dx:av_block_gt_first_degree_or_sick_sinus_without_pacemaker": FALSE}))))
    assert run(TRUE) == {"nondhp_ccb_sbp_lt_90"}
    assert run(FALSE) == set()


def test_ssri_with_anticoagulant_and_low_sodium():
    base = {"med:maoi_within_14_days": FALSE, "age_ge_65": TRUE}
    def run(anticoag, low_na):
        return set(by_id(evaluate_rules(RULESET.rules, refill(
            "ssri", **{"med:anticoagulant": anticoag, "ref:sodium_below_lln": low_na}, **base))))
    assert run(TRUE, FALSE) == {"ssri_snri_with_anticoagulant"}
    assert run(FALSE, TRUE) == {"antidepressant_low_sodium_age_ge_65"}
    assert run(FALSE, FALSE) == set()


def test_citalopram_age_and_dose():
    base = {"rx_class:ssri": TRUE, "med:maoi_within_14_days": FALSE, "med:anticoagulant": FALSE, "age_ge_65": FALSE}
    def run(dose_gt_20):
        return set(by_id(evaluate_rules(RULESET.rules, refill(
            "citalopram", age_gt_60=TRUE, **{"dose:citalopram_daily_mg_gt_20": dose_gt_20}, **base))))
    assert run(TRUE) == {"citalopram_age_gt_60_dose_gt_20"}
    assert run(FALSE) == set()


def test_gabapentin_renal_threshold():
    def run(crcl):
        return set(by_id(evaluate_rules(RULESET.rules, refill(
            "gabapentin", **{"lab:crcl_lt_60": lab(crcl, FRESH, "lt", 60), "med:opioid": FALSE}))))
    assert run(45) == {"gabapentin_crcl_lt_60"}
    assert run(60) == set()


def test_levetiracetam_le_80_boundary():
    assert {f.rule_id for f in evaluate_rules(RULESET.rules, refill("levetiracetam", **{"lab:crcl_le_80": lab(80, FRESH, "le", 80)}))} == {"levetiracetam_crcl_le_80"}
    assert evaluate_rules(RULESET.rules, refill("levetiracetam", **{"lab:crcl_le_80": lab(81, FRESH, "le", 80)})) == []


def test_symptom_rules_route_to_ordering_clinician():
    f = by_id(evaluate_rules(RULESET.rules, refill(
        "lamotrigine", **{"sx:new_skin_rash": TRUE, "adherence:supply_lapsed": FALSE})))
    assert set(f) == {"lamotrigine_new_rash"}
    assert f["lamotrigine_new_rash"].owner == Owner.ORDERING_CLINICIAN


def test_incretin_pancreatitis_holds_when_not_asked():
    preds = refill("dpp4_inhibitor", **{"rx_class:sitagliptin": TRUE, "lab:egfr_lt_45": lab(70, FRESH, "lt", 45)})
    held = by_id(evaluate_rules(RULESET.rules, preds))
    assert set(held) == {"incretin_pancreatitis_suspected"}
    assert held["incretin_pancreatitis_suspected"].unresolved_predicates == ["sx:pancreatitis_suspected"]


def test_digoxin_level_and_beers_dose():
    def run(level, age65, dose):
        return set(by_id(evaluate_rules(RULESET.rules, refill(
            "digoxin", age_ge_65=age65, **{"lab:digoxin_gt_2": lab(level, FRESH, "gt", 2.0), "dose:digoxin_daily_mg_gt_0_125": dose}))))
    assert run(2.4, FALSE, FALSE) == {"digoxin_level_gt_2"}
    assert run(1.0, TRUE, TRUE) == {"digoxin_dose_gt_0_125_age_ge_65"}
    assert run(1.0, TRUE, FALSE) == set()


def test_new_lab_recency_loinc_lists():
    # Any reported eGFR satisfies metformin's annual check; fasting glucose satisfies SGA screening.
    assert evaluate_lab_recency(RULESET.lab_recency, {"metformin"}, {"98979-8": date(2026, 3, 1)}, TODAY) == []
    assert evaluate_lab_recency(RULESET.lab_recency, {"second_generation_antipsychotic"},
                                {"1558-6": date(2026, 3, 1), "2093-3": date(2026, 3, 1)}, TODAY) == []


def test_spironolactone_contract_breaches_on_potassium_ceiling():
    c = next(c for c in CONTRACTS if c.id == "spironolactone_start_or_titration")
    start = date(2026, 10, 20)
    assert evaluate_contract(c, start, {"2823-3": 4.5}, [LabValue(loinc="2823-3", value=5.5, on=date(2026, 10, 25))], TODAY) == ContractStatus.OK
    assert evaluate_contract(c, start, {"2823-3": 4.5}, [LabValue(loinc="2823-3", value=5.6, on=date(2026, 10, 25))], TODAY) == ContractStatus.BREACH
