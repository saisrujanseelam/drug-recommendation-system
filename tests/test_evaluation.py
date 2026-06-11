from src.evaluation import evaluate_retriever, relative_improvement


def test_recall_and_mrr_on_perfect_retriever():
    queries = [
        {"query": "a", "relevant_ids": ["X1"]},
        {"query": "b", "relevant_ids": ["X2", "X3"]},
    ]

    def retrieve(q, k):
        return {"a": ["X1", "X9"], "b": ["X2", "X3", "X9"]}[q][:k]

    res = evaluate_retriever("perfect", queries, retrieve, k_max=5)
    assert res.recall_at_1 == 1.0
    assert res.recall_at_5 == 1.0
    assert res.mrr == 1.0


def test_recall_partial_credit():
    queries = [{"query": "a", "relevant_ids": ["X1", "X2", "X3"]}]

    def retrieve(q, k):
        return ["X1", "X9", "X9", "X9", "X9"][:k]

    res = evaluate_retriever("partial", queries, retrieve, k_max=5)
    assert res.recall_at_1 == 1.0 / 1
    assert res.recall_at_5 == 1.0 / 3
    assert res.mrr == 1.0


def test_relative_improvement():
    assert relative_improvement(0.75, 0.5) == 0.5
    assert relative_improvement(0.5, 0.0) == float("inf")
    assert relative_improvement(0.0, 0.0) == 0.0
