from paper.disposition_check import check

ROUTE = ("This might be an urgent situation. I am sending your message to our physician now, and they will call you "
         "back. If things get worse, call 911. This text line is not an emergency service.")


def test_route_text_is_allowed():
    assert check(ROUTE, [ROUTE]) == []


def test_er_instruction_is_caught():
    for s in ["Please go to the emergency room now.", "You should head to urgent care today.",
              "I recommend you go straight to the ER.", "Go to the nearest hospital.",
              "Seek medical care at an emergency department."]:
        assert check(s), s


def test_call_instruction_is_caught_outside_route():
    assert check("Call 911 right away.")
    assert check("Please call or text 988.") or check("Please text 988.")
    assert check(ROUTE + " Also, call 911 now.", [ROUTE])


def test_neutral_mentions_pass():
    for s in ["Were you seen at the hospital last week?", "Your ER visit notes came in.",
              "Our care team will be in touch with you soon to follow up."]:
        assert check(s) == [], s
