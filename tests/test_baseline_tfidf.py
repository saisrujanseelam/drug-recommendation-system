from src.baseline_tfidf import TfidfBaseline


def test_tfidf_returns_most_relevant_doc():
    docs = [
        "lisinopril ace inhibitor hypertension high blood pressure",
        "metformin biguanide type 2 diabetes blood glucose",
        "albuterol short acting beta agonist asthma rescue inhaler",
    ]
    tfidf = TfidfBaseline()
    tfidf.fit(docs)
    ids, scores = tfidf.search("high blood pressure medication", top_k=3)
    assert ids[0] == 0
    assert scores[0] >= scores[-1]
