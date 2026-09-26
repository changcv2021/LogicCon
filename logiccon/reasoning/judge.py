"""Isolated semantic evaluation. Only this stage receives private gold annotations."""
from .metrics import indexed
from ..io import file_hash, digest
from . import prompts
from .backends import create_backend
from .engine import Reasoner
from .runtime import (locked_run, initialize, read_events, append_event, source_fingerprint,
                      environment, model_fingerprint, atomic_json, write_jsonl)
from .schema import require, nonempty


def reference_record(gold):
    if all(gold.get(k) for k in ("target", "text_claim", "visual_evidence")):
        return {k: gold[k] for k in ("target", "text_claim", "visual_evidence")}
    mutation, frame = gold["mutation"], gold["grounded_frame"]
    objects = gold["evidence"]["objects"]
    ids = mutation["target_ids"]
    return {"target": {"description": frame["target_description"], "object_ids": ids,
                        "object_names": {i: objects[i]["name"] for i in ids if i in objects}},
            "text_claim": mutation["mutated_fact"], "visual_evidence": mutation["original_fact"],
            "statement": gold["contradictory_statement"],
            "original_statement": gold["consistent_statement"]}


def parse_judgment(row):
    require(all(type(row.get(k)) is bool for k in ("TO", "TC", "VE")), "TO/TC/VE must be boolean")
    nonempty(row.get("reason"), "reason")
    return {k: row[k] for k in ("TO", "TC", "VE", "reason")}


def run_judge(config, gold_path, prediction_path, output, judge_id, *, resume=False,
              backend_factory=create_backend, fingerprint=True):
    nonempty(judge_id, "judge_id")
    gold, predictions = indexed(gold_path), indexed(prediction_path)
    require(not predictions.keys() - gold.keys(), "Unknown prediction IDs")
    manifest = {"schema": "logiccon-judge-1", "judge_id": judge_id, "config": config.to_dict(),
                "gold_sha256": file_hash(gold_path), "predictions_sha256": file_hash(prediction_path),
                "source_sha256": source_fingerprint(), "environment": environment(),
                "model": model_fingerprint(config.backend) if fingerprint else {"software_test_only": True}}
    with locked_run(output) as root:
        initialize(root, manifest, resume)
        events = read_events(root / "events.jsonl")
        reasoner = None
        with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
            for sid, record in gold.items():
                if record["label"] != "conflicting" or events.get(sid, {}).get("status") == "ok":
                    continue
                prediction = predictions.get(sid)
                if not prediction or prediction.get("label") != "conflicting":
                    judgment = {"TO": False, "TC": False, "VE": False,
                                "reason": "Missing prediction or no conflict analysis", "judge_id": "deterministic-empty"}
                    trace = []
                else:
                    if reasoner is None:
                        reasoner = Reasoner(backend_factory(config.backend), config.method)
                    reasoner.trace = []
                    response = {k: prediction.get(k) for k in
                                ("label", "target", "text_claim", "visual_evidence", "explanation")}
                    judgment = reasoner.call("judge", prompts.JUDGE,
                                             {"standard_record": reference_record(record), "response": response},
                                             None, parse_judgment)
                    judgment["judge_id"] = judge_id
                    trace = reasoner.trace
                judgment.update({"sample_id": sid, "CP": all(judgment[k] for k in ("TO", "TC", "VE")),
                                 "prediction_sha256": digest(prediction)})
                event = {"sample_id": sid, "status": "ok", "judgment": judgment, "trace": trace}
                append_event(stream, event)
                events[sid] = event
        judgments = [e["judgment"] for e in events.values()]
        write_jsonl(root / "judgments.jsonl", judgments)
        summary = {"judged": len(judgments), "judge_id": judge_id}
        atomic_json(root / "summary.json", summary)
        return summary
