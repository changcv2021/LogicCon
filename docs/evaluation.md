# Evaluation

Inference accepts only public inputs. Gold annotations are read exclusively by evaluation commands.

## Deterministic metrics

```bash
logiccon-reason score \
  --gold /path/to/LogicCon/annotations/test.jsonl \
  --predictions runs/full/predictions.jsonl \
  --output runs/full/metrics.json
```

All metrics are fractions in `[0, 1]`:

| Metric | Meaning |
|---|---|
| All | Detection accuracy across every gold sample |
| Conf | Detection accuracy on conflicting samples |
| Type | Correct conflict label and type on conflicting samples |
| TO / TC / VE | Semantic correctness of target, textual claim and visual evidence |
| CP | TO, TC and VE are all correct for the same sample |
| Comp-Det / Comp-CP / Comp-Type | Corresponding metrics on compositional conflicting samples |

Per-type metrics and prediction/judgment coverage are also reported. Missing predictions count as errors. Compositional membership comes from gold annotations. `Avg` is left null rather than applying an unspecified cross-metric weighting.

## Independent semantic judge

Use a separate judge configuration and identity. The judge sees the standard conflict record and the final evaluated response; it does not need to receive the image or internal inference trace. Textual equivalents are accepted, but attaching a fact to the wrong target or anchor is incorrect.

```bash
export LOGICCON_JUDGE_MODEL=your-pinned-judge-model
export LOGICCON_JUDGE_BASE_URL=https://your-endpoint.example/v1
# Supply LOGICCON_JUDGE_API_KEY through the environment if the endpoint requires it.

logiccon-reason judge --config configs/reasoning/judge_http.toml \
  --gold /path/to/LogicCon/annotations/test.jsonl \
  --predictions runs/full/predictions.jsonl \
  --output runs/full/judge --judge-id independent-model-snapshot

logiccon-reason score \
  --gold /path/to/LogicCon/annotations/test.jsonl \
  --predictions runs/full/predictions.jsonl \
  --judgments runs/full/judge/judgments.jsonl \
  --output runs/full/metrics-with-judge.json
```

The paper's judge is GPT-4o. Configure the desired compatible endpoint and record its exact model snapshot. Other judges can use the same protocol. Generated judgments include a prediction hash; scoring rejects stale judgments after a prediction changes. Missing or non-conflicting responses on conflicting gold samples receive zero for the conflict-point components.

Human or external semantic judgments can also be supplied as JSONL:

```json
{"sample_id":"example-001","judge_id":"independent-reviewer","TO":true,"TC":true,"VE":false}
```

If any required judgment is missing, the corresponding aggregate semantic metric is null. No exact-string surrogate is used for semantic conflict-point scores.

## Matching the evaluation scope

Score full-split predictions against full-split gold. A four-sample smoke run should be assessed against those same four gold IDs if measuring smoke metrics; scoring it against the entire test split counts all remaining examples as missing.

Use identical inputs, backbone snapshots and judge settings for base/full and cumulative ablations. Keep the output directories and run manifests, including failed samples. Report the resolved environment and measured compute cost with the resulting metrics.
