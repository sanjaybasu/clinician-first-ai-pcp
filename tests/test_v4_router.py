"""COMPASS v4 B1/B2: quote validation, the deterministic crisis router and the structured screen call. Synthetic text."""

import json
import time

from compass_pcp.arms.screen_v4 import run_screen, screen_prompt
from compass_pcp.crisis_router import (
    NullClassifier,
    ScreenForm,
    ScreenItem,
    incomplete_form,
    parse_form,
    quote_is_valid,
    route_form,
)
from compass_pcp.policy_v4 import lexicon_hits, load_crisis_policy
from tests.conftest import FIXTURES
from tests.test_arms import FakeLLM

POLICY = load_crisis_policy(FIXTURES / "crisis_v4_placeholder.yaml")
ALL_IDS = [c.id for f in POLICY.families for c in f.criteria]


def form(text: str, **items) -> tuple[ScreenForm, list[str]]:
    """A completed form: every criterion not_mentioned except the ones given as id=dict(...)."""
    rows = [{"criterion_id": cid, "status": "not_mentioned", "current": None, "person_at_risk": "unknown", "quote": ""}
            for cid in ALL_IDS if cid not in items]
    rows += [{"criterion_id": cid, **v} for cid, v in items.items()]
    return parse_form({"items": rows}, POLICY, [text]), [text]


def route(text: str, classifier=None, **items):
    f, texts = form(text, **items)
    return route_form(f, POLICY, lexicon=lexicon_hits(" ".join(texts), POLICY), classifier=classifier)


def test_policy_is_a_marked_placeholder_with_sources():
    assert POLICY.status == "placeholder_unsigned" and POLICY.signed_by is None
    assert {f.id for f in POLICY.families} == {"suicide_self_harm", "overdose_poisoning", "acute_cardiac"}
    assert all(c.source for f in POLICY.families for c in f.criteria)


def test_quote_must_be_an_exact_substring():
    texts = ["I have been thinking  about\nending it all lately"]
    assert quote_is_valid("thinking about ending it", texts)  # whitespace runs collapse
    assert not quote_is_valid("Thinking about ending it", texts)  # case differs
    assert not quote_is_valid("thinking of ending it", texts)  # paraphrase
    assert not quote_is_valid("", texts)


def test_present_current_quoted_fires_the_signed_route():
    t = "i have pills saved up and i plan to take them all tonight"
    r = route(t, ssh_immediate_danger={"status": "present", "current": True, "person_at_risk": "patient",
                                       "quote": "i plan to take them all tonight"})
    assert r.route == "emergency_911" and r.rules_fired == ["ssh_immediate_danger"]
    assert r.instruction_text == POLICY.instructions["emergency_911"].text and not r.addressed_to_other
    assert r.page_care_team and "instr_911_v0" in r.source_ids


def test_passive_ideation_routes_to_988_and_page():
    t = "honestly sometimes i wish i could just not wake up"
    r = route(t, ssh_current_ideation={"status": "present", "current": True, "person_at_risk": "patient",
                                       "quote": "i wish i could just not wake up"})
    assert r.route == "crisis_988_page" and r.page_care_team
    assert "988" in r.instruction_text


def test_historical_statement_follows_the_normal_path():
    t = "years ago i tried to kill myself but i am doing well now"
    r = route(t, ssh_immediate_danger={"status": "present", "current": False, "person_at_risk": "patient",
                                       "quote": "years ago i tried to kill myself"})
    assert r.route == "none"


def test_unknown_currentness_sets_urgent_review():
    t = "i took a bunch of my sleeping pills on purpose"
    r = route(t, od_emergency={"status": "present", "current": None, "person_at_risk": "patient",
                               "quote": "i took a bunch of my sleeping pills on purpose"})
    assert r.route == "urgent_review" and r.rules_fired == ["od_emergency:currentness_unknown"]
    assert r.applicable_uncertainty


def test_absent_and_not_mentioned_never_set_a_floor():
    t = "i do not want to hurt myself, i just need a refill"
    r = route(t, ssh_current_ideation={"status": "absent", "current": None, "person_at_risk": "patient",
                                       "quote": "i do not want to hurt myself"})
    # the lexicon hit ("hurt myself") makes the family applicable, but nothing is uncertain or present
    assert "suicide_self_harm" in r.applicable and r.route == "none" and not r.applicable_uncertainty


def test_raw_lexicon_hit_is_a_trigger_not_a_floor():
    t = "this suicide squad movie was great"
    r = route(t)  # all not_mentioned
    assert r.applicable == {"suicide_self_harm": ["lexicon"]} and r.route == "none"


def test_uncertain_needs_an_applicable_family():
    t = "everything feels heavy lately"
    unquoted = route(t, ssh_current_ideation={"status": "uncertain", "current": None, "person_at_risk": "patient",
                                              "quote": ""})
    assert unquoted.route == "none" and not unquoted.applicable
    quoted = route(t, ssh_current_ideation={"status": "uncertain", "current": None, "person_at_risk": "patient",
                                            "quote": "everything feels heavy lately"})
    assert quoted.route == "urgent_review" and quoted.rules_fired == ["ssh_current_ideation:uncertain"]
    t2 = "i keep thinking about suicide, not sure what it means"
    by_lexicon = route(t2, ssh_current_ideation={"status": "uncertain", "current": None, "person_at_risk": "patient",
                                                 "quote": ""})
    assert by_lexicon.route == "urgent_review" and by_lexicon.applicable["suicide_self_harm"] == ["lexicon"]


def test_classifier_is_an_applicability_trigger_only():
    t = "everything feels heavy lately"
    item = {"ssh_current_ideation": {"status": "uncertain", "current": None, "person_at_risk": "patient", "quote": ""}}
    assert route(t, classifier={"suicide_self_harm": 0.2}, **item).route == "none"
    assert route(t, classifier={"suicide_self_harm": 0.9}, **item).route == "urgent_review"
    assert route(t, classifier={"suicide_self_harm": 0.99}).route == "none"  # no form evidence: never a route
    assert NullClassifier().predict(t) == {}


def test_present_without_valid_quote_is_uncertain_and_applicable():
    t = "my chest has been hurting and i am sweating"
    r = route(t, cardiac_current_presentation={"status": "present", "current": True, "person_at_risk": "patient",
                                               "quote": "chest pain with sweating"})
    assert r.route == "urgent_review" and r.rules_fired == ["cardiac_current_presentation:unverified_quote"]


def test_third_person_report_gets_the_instruction_for_the_person_at_risk():
    t = "my brother just told me he has a gun and is going to end it tonight"
    r = route(t, ssh_immediate_danger={"status": "present", "current": True, "person_at_risk": "other",
                                       "quote": "he has a gun and is going to end it tonight"})
    assert r.route == "emergency_911" and r.addressed_to_other
    assert r.instruction_text == POLICY.instructions["emergency_911"].text_other


def test_clinician_report_about_a_patient_is_routed_like_any_message():
    t = "Pt is a 54M I saw today who disclosed daily thoughts that his family would be better off without him."
    r = route(t, ssh_current_ideation={"status": "present", "current": True, "person_at_risk": "other",
                                       "quote": "daily thoughts that his family would be better off without him"})
    assert r.route == "crisis_988_page" and r.addressed_to_other
    assert r.instruction_text == POLICY.instructions["crisis_988_page"].text_other


def test_clinician_own_risk_is_addressed_to_the_writer():
    t = "I'm a nurse and I have started thinking I want to die."
    r = route(t, ssh_current_ideation={"status": "present", "current": True, "person_at_risk": "clinician_self",
                                       "quote": "I have started thinking I want to die"})
    assert r.route == "crisis_988_page" and not r.addressed_to_other


def test_poison_control_is_an_addition_below_911_only():
    t = "my toddler swallowed two of my pills and i also feel like i want to die"
    exposure = {"status": "present", "current": True, "person_at_risk": "other", "quote": "my toddler swallowed two of my pills"}
    ideation = {"status": "present", "current": True, "person_at_risk": "patient", "quote": "i also feel like i want to die"}
    both = route(t, od_exposure_below_emergency=exposure, ssh_current_ideation=ideation)
    assert both.route == "crisis_988_page" and both.additions == ["poison_control_addition"]
    assert len(both.addition_texts) == 1
    alone = route(t, od_exposure_below_emergency=exposure)
    assert alone.route == "poison_control_addition" and alone.additions == []
    t2 = "my toddler swallowed two of my pills and now he is hard to wake"
    danger = {"status": "present", "current": True, "person_at_risk": "other", "quote": "now he is hard to wake"}
    emergency = route(t2, od_exposure_below_emergency=exposure | {"quote": "my toddler swallowed two of my pills"},
                      od_emergency=danger)
    assert emergency.route == "emergency_911" and emergency.additions == [] and emergency.addition_texts == []


def test_failed_screen_is_uncertain_and_sets_the_review_floor():
    f = incomplete_form(POLICY, "timeout")
    assert all(i.status == "uncertain" and not i.completed for i in f.items.values())
    r = route_form(f, POLICY)
    assert r.route == "urgent_review" and r.rules_fired == ["screen:incomplete"] and not r.screen_completed
    r2 = route_form(f, POLICY, lexicon=lexicon_hits("i took too many pills", POLICY))
    assert "od_emergency:incomplete" in r2.rules_fired  # lexicon makes the family applicable to incomplete items


def test_parse_form_drops_bad_rows_and_keeps_missing_items_incomplete():
    f = parse_form({"items": [{"criterion_id": "made_up", "status": "present"},
                              {"criterion_id": "od_emergency", "status": "maybe"},
                              {"criterion_id": "ssh_current_ideation", "status": "absent", "person_at_risk": "x"}]},
                   POLICY, ["hello"])
    assert f.completed and "made_up" not in f.items
    assert not f.items["od_emergency"].completed
    assert f.items["ssh_current_ideation"].completed and f.items["ssh_current_ideation"].person_at_risk == "unknown"
    assert not parse_form({"nope": 1}, POLICY, ["hello"]).completed
    assert isinstance(ScreenItem(criterion_id="x"), ScreenItem)


MSG = [{"role": "user", "content": "i plan to take all my pills tonight"}]


def test_run_screen_fills_the_form_and_validates_quotes():
    reply = json.dumps({"items": [{"criterion_id": "ssh_immediate_danger", "status": "present", "current": True,
                                   "person_at_risk": "patient", "quote": "take all my pills tonight"}]})
    llm = FakeLLM({"You fill a safety form": reply})
    f = run_screen(llm, MSG, POLICY)
    assert f.completed and f.items["ssh_immediate_danger"].quote_valid
    assert not f.items["od_emergency"].completed  # omitted by the screen: incomplete, not negative
    assert all(cid in screen_prompt(POLICY) for cid in ALL_IDS)


def test_run_screen_timeout_error_and_garbage_are_incomplete():
    class Slow(FakeLLM):
        def complete(self, system, messages, **kw):
            time.sleep(0.5)
            return "{}"

    class Broken(FakeLLM):
        def complete(self, system, messages, **kw):
            raise RuntimeError("api down")

    assert run_screen(Slow({}), MSG, POLICY, timeout=0.05).error == "timeout"
    assert run_screen(Broken({}), MSG, POLICY).error == "error: RuntimeError"
    assert run_screen(FakeLLM({"You fill a safety form": "no json here"}), MSG, POLICY).error == "unreadable form"


def test_lexicon_matches_whole_words_only():
    assert lexicon_hits("I OVERDOSED last night", POLICY) == {"overdose_poisoning": ["overdosed"]}
    assert lexicon_hits("the poisonous plant", POLICY) == {}
