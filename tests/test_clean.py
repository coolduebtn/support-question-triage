from triage.clean import clean_email


def test_clean_email_strips_quotes_and_signature():
    raw = ("Hi,\nthe export fails with error 500.\n\n> older text\n> more older\n\n"
           "Thanks,\nBob\n-- \nBob Smith\nACME Corp")
    out = clean_email(raw)
    assert "older text" not in out and "ACME" not in out and "Bob Smith" not in out


def test_clean_email_strips_on_wrote_block():
    raw = "It still fails.\n\nOn Mon, 3 Mar 2025 at 10:00, Jane <j@x.org> wrote:\n> Which version?\n> thanks"
    assert clean_email(raw) == "It still fails."


def test_clean_email_keeps_question():
    q = "How do I reset the publishing config in 22.01?"
    assert clean_email(q) == q


def test_clean_email_empty_after_cleaning_returns_empty():
    assert clean_email("> only a quote\n> another") == ""
    assert clean_email("") == ""
    assert clean_email("-- \nBob\nACME") == ""
    assert clean_email("Sent from my iPhone") == ""


def test_clean_email_cuts_original_message_marker():
    raw = "Please help\n-----Original Message-----\nFrom: x\nold stuff"
    assert clean_email(raw) == "Please help"


def test_clean_email_keeps_content_after_an_early_thanks():
    raw = "Hi,\nThanks\nI get error 500 when I export the invoice report from the order manager screen every time."
    assert "error 500" in clean_email(raw)


def test_clean_email_unescapes_html_entities_before_stripping_quotes():
    raw = "It fails.\n&gt; quoted &amp;gt; text\n&gt; more"
    assert clean_email(raw) == "It fails."


def test_strong_closing_drops_name_block_and_forwarded_footer():
    raw = "I tried postgres, it works.\nThanks for the help.\n\nRegards,\nYang\n\n原始邮件\n\n发件人：x\n\n> old"
    assert clean_email(raw) == "I tried postgres, it works.\nThanks for the help."


def test_strong_closing_keeps_pasted_data_after_the_name_block():
    raw = "The entry looks wrong.\nCan you check it?\n\nRegards,\nEmad\n\n{\n  \"ACCTG_TRANS_ID\": \"10571\"\n}"
    out = clean_email(raw)
    assert "Emad" not in out and "ACCTG_TRANS_ID" in out
