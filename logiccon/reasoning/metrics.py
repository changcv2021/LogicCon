"""§A.7 metrics. Missing predictions count as errors; no string-match semantic proxy."""
from ..io import digest, jsonl
from .schema import TYPES, require


def indexed(path):
    result = {}
    for row in jsonl(path):
        require(isinstance(row, dict) and isinstance(row.get("sample_id"), str)
                and bool(row["sample_id"]), "Each record needs a nonempty sample_id")
        require(row["sample_id"] not in result, "Duplicate sample_id")
        result[row["sample_id"]] = row
    return result


def score(gold_path, prediction_path, judgment_path=None):
    gold, predictions = indexed(gold_path), indexed(prediction_path)
    judgments = indexed(judgment_path) if judgment_path else {}
    require(not predictions.keys() - gold.keys() and not judgments.keys() - gold.keys(), "Unknown sample IDs")
    for pred in predictions.values():
        require(pred.get("label") in {"consistent", "conflicting"}, "Invalid predicted label")
        if pred["label"] == "conflicting":
            require(pred.get("conflict_type") in TYPES, "Invalid predicted type")
    for sid, j in judgments.items():
        require(j.get("judge_id") and all(type(j.get(k)) is bool for k in ("TO", "TC", "VE")), "Invalid semantic judgment")
        if "prediction_sha256" in j:
            require(j["prediction_sha256"] == digest(predictions.get(sid)), "Stale semantic judgment")
    ids = list(gold)
    conf = [i for i in ids if gold[i]["label"] == "conflicting"]
    comp = [i for i in conf if gold[i]["complexity"]["label"] == "compositional"]

    def accuracy(subset, function):
        return sum(bool(function(i)) for i in subset) / len(subset) if subset else None

    def detected(i):
        return predictions.get(i, {}).get("label") == gold[i]["label"]

    def typed(i):
        return detected(i) and predictions.get(i, {}).get("conflict_type") == gold[i]["conflict_type"]

    def needs_judge(i):
        return predictions.get(i, {}).get("label") == "conflicting"

    def semantic(subset, keys):
        if any(needs_judge(i) and i not in judgments for i in subset):
            return None
        return accuracy(subset, lambda i: needs_judge(i) and all(judgments[i][k] for k in keys))

    result = {"N": len(ids), "N_conflicting": len(conf), "N_compositional": len(comp),
              "prediction_coverage": len(predictions) / len(ids) if ids else None,
              "missing_predictions": len(ids) - len(predictions),
              "All": accuracy(ids, detected), "Conf": accuracy(conf, detected), "Type": accuracy(conf, typed),
              "Comp-Det": accuracy(comp, detected), "Comp-Type": accuracy(comp, typed),
              "semantic_judgment_coverage": accuracy(conf, lambda i: i in judgments),
              "semantic_scoring_coverage": accuracy(conf, lambda i: not needs_judge(i) or i in judgments),
              "uncertain_predictions": sum(p.get("uncertain", False) for p in predictions.values())}
    for key in ("TO", "TC", "VE", "CP"):
        result[key] = semantic(conf, ("TO", "TC", "VE") if key == "CP" else (key,))
    result["Comp-CP"] = semantic(comp, ("TO", "TC", "VE"))
    result["per_type"] = {}
    for typ in sorted(TYPES):
        subset = [i for i in conf if gold[i]["conflict_type"] == typ]
        result["per_type"][typ] = {"N": len(subset), "Conf": accuracy(subset, detected),
                                   "Type": accuracy(subset, typed), "CP": semantic(subset, ("TO", "TC", "VE"))}
    result["scale"] = "fraction_0_to_1"
    result["Avg"] = None
    result["Avg_note"] = "The PDF does not unambiguously specify the averaging weights; no invented aggregate."
    return result
