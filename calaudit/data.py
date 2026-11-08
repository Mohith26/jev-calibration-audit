"""Download the published WorkflowEvals datasets and align run questions with reference labels.

Everything here reads the public Hugging Face datasets behind evals.typesafe.ai. No model is called.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

# Pinned revisions (resolved 2026-11-08). Pass revision="main" to use the latest snapshot instead.
DATASETS = {
    "invoice_processing": ("typesafeAI/evalsafe-invoice-processing", "6beeb2d2acd65c086c835022f5f4d7434114cafc"),
    "customer_service": ("typesafeAI/evalsafe-customer-service", "b1342f5a704587dbc465867c38c2694348ff86e4"),
    "security_incidents": ("typesafeAI/evalsafe-security-incidents", "fbe1ea5c69cf494157fd23f2002a0d9d9a418443"),
    "agent_trace_observability": (
        "typesafeAI/evalsafe-agent-trace-observability",
        "8635540973910a92465fe2bc53e195375aa6e1a8",
    ),
}
FILES = ("dataset.json", "data/cases.parquet", "data/questions.parquet", "data/run_results.parquet")
REFERENCES = ("consensus", "openai", "anthropic")
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data"


def fetch(workflow: str, cache: Path = DEFAULT_CACHE, revision: str | None = None) -> Path:
    """Download one workflow's files (about 1-17 MB each) into cache/<workflow>/<revision>/."""
    repo, pinned = DATASETS[workflow]
    rev = revision or pinned
    root = cache / workflow / rev
    for name in FILES:
        target = root / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
            out.write(response.read())
        tmp.rename(target)
    return root


def norm(text: str) -> str:
    """Canonical JSON so run questions and reference questions match regardless of key order."""
    return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))


def answer_of(record: dict) -> str | None:
    """The answer a model (or reference) committed to, as a string. None means a tie / no answer."""
    raw = record.get("answer_json")
    if raw is None:
        return None
    value = json.loads(raw)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def probabilities_of(record: dict) -> dict[str, float]:
    return {str(p["option"]): float(p["probability"]) for p in record.get("probabilities") or []}


@dataclass
class Answer:
    """One model's answer to one question instance, with the reference answers attached."""

    workflow: str
    model: str
    case_id: str
    key: tuple
    kind: str
    answer: str | None
    confidence: float  # probability the model put on the answer it committed to
    top_probability: float
    refs: dict[str, str | None] = field(default_factory=dict)

    def agrees(self, reference: str = "consensus") -> bool:
        return self.answer is not None and self.answer == self.refs[reference]


@dataclass
class Workflow:
    name: str
    root: Path
    manifest: dict
    answers: list[Answer]
    unmatched: dict[str, int]
    case_correct: dict[tuple[str, str], bool]  # (model, case_id) -> headline metric right for every policy
    published_accuracy: dict[str, float]  # model -> headline score reported in dataset.json
    recomputed_accuracy: dict[str, float]  # model -> headline agreement recomputed here
    comparison: str = "exact_actions"

    @property
    def models(self) -> list[str]:
        return sorted({a.model for a in self.answers})


def outcome(decision: dict, comparison: str):
    """What the headline metric compares: the set of actions, or the primary action."""
    if comparison == "exact_actions":
        return sorted((a["name"], norm(a["arguments_json"])) for a in decision["actions"])
    primary = decision.get("primary_action")
    return None if primary is None else (primary["name"], norm(primary["arguments_json"]))


def model_name(model: dict) -> str:
    return model["name"].rsplit("/", 1)[-1]


def question_key(case_id: str, q: dict) -> tuple:
    """Identity of a question instance: case, node, question id, and digests of its definition and state."""
    digest = hashlib.sha256((norm(q["question_json"]) + "\n" + norm(q["state_json"])).encode()).hexdigest()
    return (case_id, q["node_id"], q["question_id"], digest)


def load(workflow: str, cache: Path = DEFAULT_CACHE, revision: str | None = None) -> Workflow:
    root = fetch(workflow, cache, revision)
    manifest = json.loads((root / "dataset.json").read_text())
    if manifest["workflow"] != workflow:
        raise ValueError(f"{root} is not the {workflow} workflow")
    reference = {}
    for q in pq.read_table(root / "data/questions.parquet").to_pylist():
        reference[question_key(q["case_id"], q)] = {r: answer_of(q[r]) for r in REFERENCES}
    cases = {c["case_id"]: c for c in pq.read_table(root / "data/cases.parquet").to_pylist()}

    answers: list[Answer] = []
    unmatched: dict[str, int] = {}
    case_correct: dict[tuple[str, str], bool] = {}
    decisions_total: dict[str, int] = {}
    decisions_ok: dict[str, int] = {}
    published: dict[str, float] = {}
    # The workflow's headline metric: exact action sets for most, primary action for agent traces.
    headline = manifest["metrics"][0]
    comparison = headline["comparison"]
    if comparison not in ("exact_actions", "primary_action"):
        raise ValueError(f"Unsupported comparison {comparison!r}")
    for run_id, run in manifest["runs"].items():
        score = next(s for s in run["scores"] if s["metric_id"] == headline["metric_id"])
        published[model_name(run["model"])] = score["value"]

    for row in pq.read_table(root / "data/run_results.parquet").to_pylist():
        name = model_name(row["model"])
        # Case-level correctness: the headline metric matches the consensus label for every policy.
        ref_decisions = {d["policy_id"]: d for d in cases[row["case_id"]]["consensus"]["decisions"]}
        ok_all = bool(row["decisions"])
        for decision in row["decisions"]:
            ref = ref_decisions.get(decision["policy_id"])
            ok = decision["status"] == "ok" and ref is not None and outcome(decision, comparison) == outcome(
                ref, comparison
            )
            decisions_total[name] = decisions_total.get(name, 0) + 1
            decisions_ok[name] = decisions_ok.get(name, 0) + int(ok)
            ok_all = ok_all and ok
        if not row["decisions"]:
            # A failed case still counts against the run, once per policy (as the published scores do).
            decisions_total[name] = decisions_total.get(name, 0) + len(manifest["policies"])
        case_correct[(name, row["case_id"])] = ok_all

        for q in row["questions"] or []:
            if q["status"] != "answered":
                continue
            key = question_key(row["case_id"], q)
            if key not in reference:
                unmatched[name] = unmatched.get(name, 0) + 1
                continue
            probs = probabilities_of(q)
            if not probs:
                continue
            ans = answer_of(q)
            conf = max(probs.values()) if ans is None else probs.get(ans)
            if conf is None:
                raise ValueError(f"{name} answered {ans!r} but gave no probability for it: {key[:3]}")
            answers.append(
                Answer(
                    workflow=workflow,
                    model=name,
                    case_id=row["case_id"],
                    key=key,
                    kind=q["kind"],
                    answer=ans,
                    confidence=conf,
                    top_probability=max(probs.values()),
                    refs=reference[key],
                )
            )
    recomputed = {m: decisions_ok.get(m, 0) / decisions_total[m] for m in decisions_total}
    return Workflow(workflow, root, manifest, answers, unmatched, case_correct, published, recomputed, comparison)
