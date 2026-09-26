"""Small deterministic software fixtures, never benchmark/model performance evidence."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from logiccon.reasoning.backends import HTTPBackend, Response
from logiccon.reasoning.config import BackendConfig, RunConfig, load_config
from logiccon.reasoning.engine import MethodConfig, ModelOutputError, Reasoner
from logiccon.reasoning.judge import reference_record, run_judge
from logiccon.reasoning.metrics import score
from logiccon.reasoning.runtime import merge_runs, read_events, run_inference, write_jsonl
from logiccon.reasoning.schema import Sample, parse_claims, parse_json


CLAIM = {"claim_id": "c1", "target": "the cup left of the plate", "predicate": "color",
         "detail": "red", "text_claim": "The cup left of the plate is red.", "context": ["left of the plate"]}
PARSED = {"claims": [CLAIM], "compositional": True}


def verification(verdict="contradicted", cid="c1"):
    return {"claim_id": cid, "target": CLAIM["target"], "visual_evidence": "The cup is blue.",
            "verdict": verdict, "anchors": ["plate"], "subclaims": [],
            "missing_evidence": "The cup is obscured." if verdict == "insufficient" else ""}


def diagnosis(label="conflicting", uncertain=False, conflict_type="attribute"):
    return {"label": label, "primary_claim_id": "c1" if label == "conflicting" else None,
            "conflict_type": conflict_type if label == "conflicting" else None,
            "target": CLAIM["target"] if label == "conflicting" else "",
            "text_claim": CLAIM["text_claim"] if label == "conflicting" else "",
            "visual_evidence": "The cup is blue." if label == "conflicting" else "",
            "uncertain": uncertain, "explanation": "The localized cup is blue."}


class ScriptedBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, image=None):
        self.calls.append((prompt, image))
        row = next(self.responses)
        if isinstance(row, Exception):
            raise row
        return Response(row if isinstance(row, str) else json.dumps(row), {"input_tokens": 10, "output_tokens": 5})


class ReasoningTests(unittest.TestCase):
    sample = Sample("fixture", "synthetic.jpg", CLAIM["text_claim"])

    def run_script(self, responses, **options):
        backend = ScriptedBackend(responses)
        reasoner = Reasoner(backend, MethodConfig(**options))
        return reasoner.run(self.sample), backend, reasoner

    def test_four_modules_and_early_stop(self):
        result, backend, reasoner = self.run_script([PARSED, verification(), diagnosis(), diagnosis()])
        self.assertEqual([r["stage"] for r in reasoner.trace], ["parse", "verify", "aggregate", "final"])
        self.assertIsNone(backend.calls[0][1])
        self.assertTrue(all(c[1] == "synthetic.jpg" for c in backend.calls[1:]))
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(result["usage"], {"input_tokens": 40, "output_tokens": 20})
        self.assertEqual(result["conflict_points"][0]["claim_id"], "c1")

    def test_k_three_means_total_three_rounds(self):
        uncertain = diagnosis("consistent", True)
        result, _, reasoner = self.run_script([PARSED] + [verification("insufficient")] * 3 + [uncertain] * 2)
        self.assertEqual([r["stage"] for r in reasoner.trace], ["parse", "verify", "refine", "refine", "aggregate", "final"])
        self.assertEqual(len(result["verification_history"]["c1"]), 3)
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["evidence_status"], "insufficient")

    def test_refinement_recovers_and_preserves_original_claim(self):
        result, backend, _ = self.run_script([PARSED, verification("insufficient"), verification(), diagnosis(), diagnosis()])
        self.assertEqual(len(result["verification_history"]["c1"]), 2)
        refine_payload = json.loads(backend.calls[2][0].split("TASK_DATA:\n")[1])
        self.assertEqual(refine_payload["claim"], CLAIM)
        self.assertEqual(len(refine_payload["previous_verifications"]), 1)

    def test_all_supported(self):
        result, _, _ = self.run_script([PARSED, verification("supported"), diagnosis("consistent"), diagnosis("consistent")])
        self.assertEqual(result["label"], "consistent")
        self.assertEqual(result["evidence_status"], "supported")
        self.assertEqual(result["conflict_points"], [])

    def test_any_contradiction_wins_with_multiple_claims(self):
        second = {**CLAIM, "claim_id": "c2"}
        parsed = {"claims": [CLAIM, second], "compositional": True}
        result, _, _ = self.run_script([parsed, verification(), verification("supported", "c2"), diagnosis(), diagnosis()])
        self.assertEqual(len(result["claims"]), 2)
        self.assertEqual(result["label"], "conflicting")

    def test_type_comes_from_diagnosis_not_predicate_keyword(self):
        parsed = copy.deepcopy(PARSED)
        parsed["claims"][0]["predicate"] = "holding"
        result, _, _ = self.run_script([parsed, verification(), diagnosis(conflict_type="object"), diagnosis(conflict_type="object")])
        self.assertEqual(result["conflict_type"], "object")

    def test_final_drift_rejected_not_silently_scored(self):
        with self.assertRaises(ModelOutputError):
            self.run_script([PARSED, verification(), diagnosis(), diagnosis("consistent")], format_retries=0)

    def test_aggregation_cannot_ignore_contradiction(self):
        with self.assertRaises(ModelOutputError):
            self.run_script([PARSED, verification(), diagnosis("consistent")], format_retries=0)

    def test_decisive_evidence_required(self):
        record = verification()
        record["visual_evidence"] = ""
        with self.assertRaises(ModelOutputError):
            self.run_script([PARSED, record], format_retries=0)

    def test_invalid_json_repair_is_traced(self):
        result, _, reasoner = self.run_script(["not JSON", PARSED, verification(), diagnosis(), diagnosis()])
        self.assertEqual(result["model_calls"], 5)
        self.assertIn("validation_error", reasoner.trace[0])

    def test_ablation_stage_counts(self):
        cases = {"base": ([diagnosis()], 1), "parsing": ([PARSED, diagnosis()], 2),
                 "verification": ([PARSED, verification(), diagnosis()], 3),
                 "aggregation": ([PARSED, verification(), diagnosis()], 3)}
        for mode, (responses, count) in cases.items():
            with self.subTest(mode=mode):
                result, _, reasoner = self.run_script(responses, mode=mode)
                self.assertEqual(result["model_calls"], count)
                self.assertNotIn("final", [r["stage"] for r in reasoner.trace])

    def test_empty_duplicate_or_truncated_claims_rejected(self):
        for claims in ([], [CLAIM, CLAIM]):
            with self.assertRaises(ValueError):
                parse_claims({"claims": claims, "compositional": False}, 16)
        with self.assertRaises(ValueError):
            parse_claims({"claims": [CLAIM, {**CLAIM, "claim_id": "c2"}], "compositional": False}, 1)

    def test_json_not_scraped_from_arbitrary_prose(self):
        self.assertEqual(parse_json('```json\n{"ok": true}\n```'), {"ok": True})
        for raw in ('Explanation {"ok":true}', '{"ok": true, "ok": false}', '[]'):
            with self.assertRaises(ValueError):
                parse_json(raw)

    def test_input_annotations_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fixture.jpg").write_bytes(b"synthetic")
            row = {"sample_id": "test", "image": "fixture.jpg", "statement": "Text"}
            self.assertEqual(Sample.from_dict(row, root).sample_id, "test")
            for changed in ({**row, "label": "conflicting"}, {**row, "image": "../fixture.jpg"}):
                with self.assertRaises(ValueError):
                    Sample.from_dict(changed, root)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "fixture.jpg").write_bytes(b"SOFTWARE_TEST_NOT_REAL_IMAGE")
        self.inputs = self.root / "inputs.jsonl"
        write_jsonl(self.inputs, [{"sample_id": str(i), "image": "fixture.jpg", "statement": "test"} for i in range(3)])
        self.config = RunConfig(BackendConfig(kind="http", model="fixture", base_url="https://example.invalid/v1"),
                                MethodConfig(mode="base", format_retries=0))

    def run_fake(self, responses, output=None, **kwargs):
        backend = ScriptedBackend(responses)
        with patch("sys.stdout", new=io.StringIO()):
            result = run_inference(self.config, self.inputs, self.root, output or self.root / "run",
                                   backend_factory=lambda _: backend, fingerprint=False, **kwargs)
        return result, backend

    def test_completed_resume_never_calls_model(self):
        self.run_fake([diagnosis()] * 3)
        result, backend = self.run_fake([], resume=True)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(backend.calls, [])

    def test_changed_image_config_and_inputs_refuse_resume(self):
        self.run_fake([diagnosis()] * 3)
        (self.root / "fixture.jpg").write_bytes(b"CHANGED")
        with self.assertRaisesRegex(ValueError, "image/sample changed"):
            self.run_fake([], resume=True)
        self.config = RunConfig(self.config.backend, MethodConfig(mode="full"))
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.run_fake([], resume=True)

    def test_failure_saved_and_retry_on_resume(self):
        with self.assertRaises(RuntimeError):
            self.run_fake([diagnosis(), RuntimeError("simulated interruption")])
        saved = read_events(self.root / "run/events.jsonl")
        self.assertEqual(saved["0"]["status"], "ok")
        self.assertEqual(saved["1"]["status"], "error")
        result, backend = self.run_fake([diagnosis()] * 2, resume=True)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(len(backend.calls), 2)

    def test_partial_tail_preserved_and_recovered(self):
        self.run_fake([diagnosis()] * 3)
        log = self.root / "run/events.jsonl"
        with log.open("ab") as f:
            f.write(b'{"partial":')
        result, _ = self.run_fake([], resume=True)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(len(list(log.parent.glob("events.jsonl.partial-*"))), 1)

    def test_shards_cover_once_and_merge_requires_all(self):
        self.run_fake([diagnosis()] * 2, self.root / "s0", num_shards=2, shard_index=0)
        self.run_fake([diagnosis()], self.root / "s1", num_shards=2, shard_index=1)
        with self.assertRaisesRegex(ValueError, "every shard"):
            merge_runs([self.root / "s0"], self.root / "partial.jsonl")
        result = merge_runs([self.root / "s0", self.root / "s1"], self.root / "merged.jsonl")
        self.assertEqual(result, {"samples": 3, "shards": 2})

    def test_limit_before_sharding(self):
        result, _ = self.run_fake([diagnosis()], limit=2, num_shards=2, shard_index=1)
        self.assertEqual(result["selected"], 1)

    def test_keep_going_does_not_fabricate_prediction(self):
        result, _ = self.run_fake([RuntimeError("failure"), diagnosis(), diagnosis()], keep_going=True)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["completed"], 2)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gold, self.pred, self.judged = [self.root / x for x in ("gold.jsonl", "pred.jsonl", "judged.jsonl")]
        write_jsonl(self.gold, [{"sample_id": "a", "label": "conflicting", "conflict_type": "attribute",
                                "complexity": {"label": "compositional"}, "target": "cup", "text_claim": "red",
                                "visual_evidence": "blue"},
                               {"sample_id": "b", "label": "consistent", "conflict_type": None,
                                "complexity": {"label": "atomic"}}])
        write_jsonl(self.pred, [{"sample_id": "a", **diagnosis()}, {"sample_id": "b", **diagnosis("consistent")}])

    def test_semantic_metrics_require_judge(self):
        result = score(self.gold, self.pred)
        self.assertEqual(result["All"], 1)
        self.assertEqual(result["Comp-Type"], 1)
        self.assertIsNone(result["CP"])
        self.assertIsNone(result["Avg"])

    def test_cp_requires_every_component(self):
        write_jsonl(self.judged, [{"sample_id": "a", "judge_id": "independent", "TO": True, "TC": True, "VE": False}])
        result = score(self.gold, self.pred, self.judged)
        self.assertEqual(result["TO"], 1)
        self.assertEqual(result["CP"], 0)
        self.assertEqual(result["Comp-CP"], 0)

    def test_missing_predictions_are_errors_not_dropped(self):
        write_jsonl(self.pred, [])
        result = score(self.gold, self.pred)
        self.assertEqual(result["All"], 0)
        self.assertEqual(result["CP"], 0)
        self.assertEqual(result["missing_predictions"], 2)

    def test_independent_judge_output_and_staleness(self):
        config = RunConfig(BackendConfig(kind="http", model="judge", base_url="https://example.invalid/v1"))
        backend = ScriptedBackend([{"TO": True, "TC": True, "VE": True, "reason": "Same facts"}])
        run_judge(config, self.gold, self.pred, self.root / "judge", "fixture-judge",
                  backend_factory=lambda _: backend, fingerprint=False)
        judgments = self.root / "judge/judgments.jsonl"
        self.assertEqual(score(self.gold, self.pred, judgments)["CP"], 1)
        self.assertIsNone(backend.calls[0][1])
        write_jsonl(self.pred, [{"sample_id": "a", **diagnosis("consistent")}])
        with self.assertRaisesRegex(ValueError, "Stale"):
            score(self.gold, self.pred, judgments)


class ConfigAndHTTPTests(unittest.TestCase):
    def test_unknown_config_keys_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text('[backend]\nmodel="example"\nunknown=true\n')
            with self.assertRaises(ValueError):
                load_config(path)

    def test_http_payload_and_token_accounting_without_network(self):
        config = BackendConfig(kind="http", model="fixture", base_url="https://example.invalid/v1")
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "{}"}}],
                                         "usage": {"prompt_tokens": 12, "completion_tokens": 2}}).encode())
        with patch("urllib.request.urlopen", return_value=response) as call:
            result = HTTPBackend(config).generate("test")
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "test")
        self.assertEqual(result.usage, {"input_tokens": 12, "output_tokens": 2})


if __name__ == "__main__":
    unittest.main()
