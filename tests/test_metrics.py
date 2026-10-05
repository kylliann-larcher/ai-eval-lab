import math

import pytest

from anomalies.accuracy_trap import evaluate, run
from rag_eval.evaluate import run as run_rag
from rag_eval.judge import agreement
from rag_eval.metrics import compare_runs, keywords_match, recall_at_k, reciprocal_rank


def test_recall_at_k():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, "a", 1) == 1.0
    assert recall_at_k(ranked, "b", 1) == 0.0
    assert recall_at_k(ranked, "b", 3) == 1.0
    assert recall_at_k(ranked, "z", 3) == 0.0


def test_reciprocal_rank():
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, "a") == 1.0
    assert reciprocal_rank(ranked, "b") == 0.5
    assert reciprocal_rank(ranked, "z") == 0.0


def test_keywords_match_ignores_case_and_accents():
    assert keywords_match("La PRÉCISION est faible", ["precision"])
    assert keywords_match("entre 30 à 50 questions", ["30 à 50"])
    assert not keywords_match("le rappel", ["rappel", "précision"])


def test_compare_runs():
    before = {"q1": True, "q2": False, "q3": True}
    after = {"q1": False, "q2": True, "q3": True}
    assert compare_runs(before, after) == {"regressions": ["q1"], "fixes": ["q2"]}


def test_agreement():
    a = agreement([True, False, True, True], [True, False, False, True])
    assert a["raw_agreement"] == 0.75
    assert 0 < a["cohen_kappa"] < 1
    assert math.isnan(agreement([True, True], [True, True])["cohen_kappa"])
    with pytest.raises(ValueError):
        agreement([True], [True, False])


def test_evaluate_counts_missed_anomalies():
    r = evaluate([0, 0, 0, 1], [0, 0, 0, 0])
    assert r["accuracy"] == 0.75
    assert r["recall"] == 0.0
    assert r["confusion_matrix"] == [[3, 0], [1, 0]]


def test_accuracy_trap():
    """Le modèle « toujours normal » a ~99 % d'accuracy et 0 % de rappel."""
    results = run()
    dummy = results["toujours_normal"]
    assert dummy["accuracy"] == pytest.approx(0.99, abs=0.005)
    assert dummy["recall"] == 0.0
    assert results["gradient_boosting"]["recall"] > 0.5


@pytest.mark.parametrize("retriever", ["words", "chars"])
def test_rag_eval_runs(retriever):
    summary = run_rag(retriever)["summary"]
    assert summary["n_questions"] == 24
    assert 0 <= summary["recall@1"] <= summary["recall@3"] <= 1
