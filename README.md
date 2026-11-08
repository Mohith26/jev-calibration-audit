# jev-calibration-audit

An independent check of how well Jev's probabilities are calibrated, using only the per-question data TypeSafe published with [WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals) (the runs behind [evals.typesafe.ai](https://evals.typesafe.ai)).

WorkflowEvals scores each model on final action agreement and cost. Every published run also stores the probability distribution each model gave for every yes/no, choice and score question it was asked: 9 models (Jev 1.13.0 plus eight LLMs), 4 workflows, about 11,000 to 12,000 categorical questions per model. Jev's pitch is calibrated, typed decisions that code can branch on, so I wanted to know whether those probabilities are good enough to gate automation on, and how they compare with the LLMs' verbalized probabilities on exactly the same workflow questions.

No model is called. Nothing here needs an API key. Everything is recomputed from the public Hugging Face datasets at pinned revisions.

## What I found

All numbers are agreement with the WorkflowEvals **consensus reference** (the mean of the OpenAI and Anthropic reference models' probabilities), not human ground truth. Full tables: [results/RESULTS.md](results/RESULTS.md).

1. **The published scores reproduce exactly.** Recomputing exact action-set agreement (primary-action agreement for agent traces) from the published decisions matches the score in every dataset's `dataset.json` for all 36 runs.

2. **Jev's question-level calibration is among the best of the nine, and it errs on the side of underconfidence.** Pooled ECE (10 bins) is 0.029 for Jev, second to gpt-5.6-sol (0.025) and ahead of claude-sonnet-5 (0.031), deepseek-v4-pro (0.033) and claude-opus-5 (0.049). Jev's mean confidence is 89.5% against 92.2% agreement (2.7 points underconfident). The GPT-5.6 and DeepSeek runs are 2.5 to 5.0 points overconfident; claude-haiku-4-5 is 12.2 points overconfident. Restricting every model to the 10,954 question instances all nine runs answered gives the same ordering for Jev (ECE 0.030).

3. **On contested questions, Jev says so.** On questions where the two reference models disagree with each other, Jev's mean confidence is 74.0% and only 4.4% of its answers are at 0.99 or above, close to claude-opus-5 and claude-sonnet-5 (72 to 74%, 1 to 2%). The GPT-5.6 and DeepSeek runs put 0.99 or more on 21 to 46% of those questions.

4. **That difference decides whether a high-agreement gate is possible at all.** Learn-then-Test (exact binomial test, delta 0.1, thresholds chosen on half the cases and evaluated on the other half, 200 splits) can certify a threshold for 98% agreement for only four models: claude-opus-5 accepts 79.3% of held-out questions, gpt-5.6-sol 72.7%, **Jev 65.8%**, claude-sonnet-5 62.5%. For gpt-5.6-luna, gpt-5.6-terra, both DeepSeek runs and claude-haiku-4-5 no threshold reaches 98%, because 59 to 82% of their answers sit tied at the same top confidence (usually exactly 1.0) and that tie as a whole agrees less than 98% of the time. A threshold cannot split a tie. At a 95% target, Jev accepts 90.0% (opus 94.8%, sol 95.3%).

5. **Good question calibration does not carry over to whole cases.** Gating a case on its least confident answer lets few cases through at 95% case accuracy for any model. Jev is best on customer service (16.7% of cases, in-sample) but lets through 0.7% on invoice processing, where its case accuracy is 58.0% against 74.7% for claude-opus-5 and 76.7% for gpt-5.6-sol. With 111 to 240 cases per workflow these case-level numbers are in-sample only.

6. **Reference family matters a little.** GPT-5.6 and DeepSeek runs agree 1.1 to 2.0 points more with the OpenAI reference than with the Anthropic one; Claude runs and Jev lean 0.7 to 0.8 points the other way. Small, but the same size as some leaderboard gaps.

## What this is not

- Agreement with a two-model consensus is not correctness. A model that is right where both references are wrong is scored as wrong here.
- I did not call Jev or any LLM. I cannot say anything about Jev versions other than 1.13.0, about latency, or about cost beyond what the datasets record.
- The LTT guarantee assumes calibration and test questions are exchangeable and treats questions as independent. I split by case so a case's questions stay on one side, but questions inside a case are still correlated, so read the guarantee as approximate. The `below target` column reports how often held-out agreement actually fell below the target (at most 5% of splits for any model at either target).
- Each model is scored on the questions its own workflow path asked. Section 2 of the results repeats the main numbers on the common question set.
- Score-type questions (0 to N levels) are excluded from the calibration tables and only used in the case-level gate.

## Run it

Python 3.12+.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # 18 tests; downloads ~23 MB of public data on first run
.venv/bin/python run_all.py            # writes results/RESULTS.md and CSVs (about 1 minute on Apple Silicon arm64)
.venv/bin/python run_all.py --quick    # 20 splits instead of 200
```

Data is cached in `data/` (gitignored). The dataset revisions are pinned in `calaudit/data.py`.

## Layout

- `calaudit/data.py`: downloads the datasets, matches each run's questions to the reference questions by case, node, question id and a digest of the canonical question and state JSON, and recomputes the published headline scores.
- `calaudit/metrics.py`: ECE, Brier, reliability bins, tie-aware risk-coverage, Learn-then-Test thresholds and repeated case-level splits.
- `run_all.py`: builds every table in `results/`.
- `tests/`: unit tests for the metrics (including a synthetic check that LTT keeps risk under alpha) and integration tests on the real data (every invoice question matched, all 36 published scores reproduced, consensus answers equal the argmax of the mean reference distribution).

Data: TypeSafe's WorkflowEvals datasets on Hugging Face (`typesafeAI/evalsafe-*`). Code in this repo: MIT.
