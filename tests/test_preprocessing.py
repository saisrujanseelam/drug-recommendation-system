from src.preprocessing import normalize_text, build_document


def test_normalize_text_lowercases_and_collapses_whitespace():
    assert normalize_text("  Foo   Bar\nBaz ") == "foo bar baz"


def test_normalize_text_preserves_medical_tokens():
    out = normalize_text("Lisinopril 10 mg/day - 5% reduction")
    assert "mg/day" in out
    assert "5%" in out


def test_normalize_text_handles_empty():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_build_document_includes_indications_twice():
    row = {
        "name": "Foo",
        "generic_name": "foo",
        "drug_class": "Class",
        "indications": "hypertension",
        "description": "blood pressure med",
        "side_effects": "cough",
    }
    doc = build_document(row)
    assert doc.count("hypertension") == 2
    assert "blood pressure med" in doc
