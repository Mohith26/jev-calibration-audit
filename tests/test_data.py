"""Integration tests against the real published datasets (downloads about 23 MB on first run)."""

import json

import pytest

from calaudit.data import DATASETS, answer_of, load, norm, probabilities_of

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def workflows():
    return {name: load(name) for name in DATASETS}


def test_answer_parsing():
    assert answer_of({"answer_json": "true"}) == "true"
    assert answer_of({"answer_json": "false"}) == "false"
    assert answer_of({"answer_json": '"approves_as_billed"'}) == "approves_as_billed"
    assert answer_of({"answer_json": "3"}) == "3"
    assert answer_of({"answer_json": None}) is None
    assert probabilities_of({"probabilities": [{"option": "true", "probability": 0.7}]}) == {"true": 0.7}


def test_norm_ignores_key_order():
    assert norm('{"b": 1, "a": [1, {"d": 2, "c": 3}]}') == norm('{"a":[1,{"c":3,"d":2}],"b":1}')


def test_published_headline_scores_reproduce_exactly(workflows):
    for name, wf in workflows.items():
        assert set(wf.published_accuracy) == set(wf.recomputed_accuracy), name
        for model, value in wf.published_accuracy.items():
            assert wf.recomputed_accuracy[model] == pytest.approx(value, abs=1e-12), (name, model)


def test_every_invoice_question_matches_a_reference(workflows):
    wf = workflows["invoice_processing"]
    assert wf.unmatched == {}
    assert len(wf.models) == 9
    for model in wf.models:
        assert sum(1 for a in wf.answers if a.model == model) == wf.manifest["coverage"]["questions"]


def test_confidence_is_probability_of_committed_answer(workflows):
    for wf in workflows.values():
        for a in wf.answers[:5000]:
            assert 0.0 <= a.confidence <= a.top_probability + 1e-12
            if a.answer is None:
                assert a.confidence == a.top_probability


def test_consensus_reference_matches_mean_of_contributors():
    # The consensus answer must be the argmax of the mean of the two reference distributions.
    import pyarrow.parquet as pq

    wf = load("customer_service")
    rows = pq.read_table(wf.root / "data/questions.parquet").to_pylist()[:500]
    for q in rows:
        if q["kind"] == "score" or q["consensus"]["answer_json"] is None:
            continue
        o, a = probabilities_of(q["openai"]), probabilities_of(q["anthropic"])
        if not o or not a:
            continue
        mean = {k: (o.get(k, 0) + a.get(k, 0)) / 2 for k in set(o) | set(a)}
        best = max(mean.values())
        winners = {k for k, v in mean.items() if abs(v - best) < 1e-9}
        assert answer_of(q["consensus"]) in winners or json.loads(q["consensus"]["answer_json"]) in winners
