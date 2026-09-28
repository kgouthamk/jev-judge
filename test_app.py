from decimal import Decimal

import pytest

import app


def crit(**kw):
    return app.new_criterion(statement=kw.pop("statement", "x"), **kw)


# ── parsing ──────────────────────────────────────────────────────────────────


def test_binary_must_not_have_inverts():
    c = crit(polarity="must_not_have")
    a = app.parse_answer(c, {"type": "noul", "noul": 0.91})
    assert a["fraction"] == 0.0 and a["confidence"] == pytest.approx(0.91)
    a = app.parse_answer(c, {"type": "noul", "noul": 0.08})
    assert a["fraction"] == 1.0 and a["confidence"] == pytest.approx(0.92)


def test_choice_uses_acceptable_set_and_falls_back_to_argmax():
    c = crit(kind="choice", options_text="Formal: a\nFriendly: b\nRude: c", acceptable=["Formal", "Friendly"])
    a = app.parse_answer(c, {"choice": "Friendly", "probabilities": {"Formal": 0.1, "Friendly": 0.8, "Rude": 0.1}, "confidence": 0.8})
    assert a["fraction"] == 1.0 and a["confidence"] == 0.8
    a = app.parse_answer(c, {"choice": "???", "probabilities": {"Formal": 0.1, "Friendly": 0.2, "Rude": 0.7}})
    assert a["label"].endswith("Rude") and a["fraction"] == 0.0 and a["confidence"] == 0.7


@pytest.mark.parametrize("raw", [
    {"score": "4 - Good"},
    {"score": 3},
    {"score": 3.2},
    {"probabilities": {"3": 0.9, "0": 0.1}},
])
def test_score_accepts_label_index_or_probabilities(raw):
    c = crit(kind="score")
    a = app.parse_answer(c, raw)
    assert a["fraction"] == pytest.approx(0.75)


def test_live_score_shape_expected_value_with_index_probabilities():
    # Captured from a real jev-1.13-free response.
    raw = {"type": "score", "score": 3.93, "confidence": 0.94,
           "probabilities": {"0": 0, "1": 0, "2": 0, "3": 0.06, "4": 0.94}}
    a = app.parse_answer(crit(kind="score"), raw)
    assert a["fraction"] == 1.0 and a["label"].startswith("5 of 5") and a["confidence"] == 0.94


def test_savings_pct_and_label():
    c = app.compute_costs(939, 69)
    assert app.savings_pct(c["jev"], c["frontier"]) == pytest.approx(99.69, abs=0.01)
    assert "(99.7%) vs frontier LLM judge" in app.fmt_savings(c["jev"], c["frontier"])
    assert app.savings_pct(Decimal(0), Decimal(0)) == 0.0


def test_missing_or_malformed_answers_become_errors():
    assert app.parse_answer(crit(), None)["error"]
    assert app.parse_answer(crit(), {"noul": "yes"})["error"]
    assert app.parse_answer(crit(kind="score"), {"score": "nope"})["error"]


# ── grading ──────────────────────────────────────────────────────────────────


def _plan_example():
    pii = crit(polarity="must_not_have", importance="High", deal_breaker=True)
    apology = crit(importance="Medium")
    tone = crit(kind="choice", importance="Medium", options_text="Formal\nNeutral\nCasual", acceptable=["Formal", "Neutral"])
    helpful = crit(kind="score", importance="High")
    return [pii, apology, tone, helpful]


def test_plan_worked_example_scores_92_5_and_passes():
    cs = _plan_example()
    answers = app.parse_response(cs, {"answers": {
        "c1": {"noul": 0.02}, "c2": {"noul": 0.97},
        "c3": {"choice": "Neutral", "confidence": 0.9}, "c4": {"score": 3, "confidence": 0.9},
    }})
    g = app.grade(cs, answers, pass_mark=90, review_conf=85)
    assert g["score"] == pytest.approx(92.5) and g["verdict"] == "PASS"


def test_confident_deal_breaker_failure_fails_despite_high_score():
    cs = _plan_example()
    answers = app.parse_response(cs, {"answers": {
        "c1": {"noul": 0.95}, "c2": {"noul": 0.97},
        "c3": {"choice": "Formal", "confidence": 0.9}, "c4": {"score": 4, "confidence": 0.9},
    }})
    g = app.grade(cs, answers, 90, 85)
    assert g["verdict"] == "FAIL" and "deal-breaker" in g["why"]


def test_uncertainty_only_triggers_review_when_it_could_flip_the_verdict():
    cs = [crit(importance="High"), crit(importance="Low")]
    # Certain High criterion = 75%; the uncertain Low one decides between 75% and 100%.
    answers = app.parse_response(cs, {"answers": {"c1": {"noul": 0.99}, "c2": {"noul": 0.6}}})
    assert app.grade(cs, answers, 90, 85)["verdict"] == "NEEDS REVIEW"
    # With a 70% pass mark, both outcomes pass → no review needed.
    assert app.grade(cs, answers, 70, 85)["verdict"] == "PASS"


def test_missing_answer_needs_review():
    cs = [crit()]
    assert app.grade(cs, app.parse_response(cs, {}), 90, 85)["verdict"] == "NEEDS REVIEW"


def test_custom_weight_is_clamped():
    assert app.criterion_weight(crit(importance="Custom", custom_weight=50)) == app.CUSTOM_MAX
    assert app.criterion_weight(crit(importance="Custom", custom_weight=0)) == app.CUSTOM_MIN


# ── payload, costs, secrets, csv ─────────────────────────────────────────────


def test_payload_maps_types_to_jev_question_types():
    rubric = {"context": "ctx", "exceptions": "exc", "criteria": _plan_example()}
    p = app.build_payload(rubric, "hello")
    assert p["model"] == "jev-1.13-free" and p["state"] == {"text": "hello"}
    assert [q["type"] for q in p["questions"].values()] == ["noul", "noul", "choice", "score"]
    assert isinstance(p["questions"]["c4"]["criteria"], list)
    assert "exc" in p["questions"]["c1"]["instructions"]


def test_costs_output_free_for_jev():
    c = app.compute_costs(1_000_000, 1_000_000)
    assert c["jev"] == Decimal("0.042")
    assert c["frontier"] == Decimal("60")
    assert c["savings"] == Decimal("59.958")


def test_fmt_usd_tiny_values():
    assert app.fmt_usd(Decimal("0.0000099")) == "$0.0000099"
    assert app.fmt_usd(Decimal("12.5")) == "$12.50"
    assert app.fmt_usd(Decimal(0)) == "$0.00"


def test_scrub_removes_keys():
    msg = "bad key sk_live_abcdef123456 sent as Bearer sk_live_abcdef123456"
    out = app.scrub(msg, "sk_live_abcdef123456")
    assert "abcdef" not in out


def test_csv_parsing_handles_bom_case_dupes_and_blanks():
    data = "﻿ID , Text_To_Evaluate\n1,hello\n1,world\n2,\n,no id\n".encode()
    df, warnings = app.parse_batch_csv(data)
    assert list(df["id"]) == ["1", "1-2", "row-4"]
    assert len(warnings) == 3


def test_csv_missing_column_raises():
    with pytest.raises(ValueError, match="text_to_evaluate"):
        app.parse_batch_csv(b"id,text\n1,hi\n")


def test_cached_repeat_costs_nothing_and_matches_ledger():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=60).run()  # no key → demo mode
    at.text_area(key="single_text").set_value("Same text twice.").run()
    evaluate = lambda: next(b for b in at.button if "Evaluate" in b.label).click().run()
    evaluate()
    first = at.session_state.single_result
    evaluate()
    second = at.session_state.single_result
    assert second["source"] == "cached" and second["costs"]["jev"] == 0
    totals = app.ledger_totals(at.session_state.ledger)
    assert totals["calls"] == 1
    assert totals["jev"] == first["costs"]["jev"] + second["costs"]["jev"]


def test_demo_response_parses_cleanly():
    rubric = {"context": "", "exceptions": "", "criteria": _plan_example()}
    body = app.demo_response(app.build_payload(rubric, "some text"))
    answers = app.parse_response(rubric["criteria"], body)
    assert all(a["error"] is None for a in answers.values())
