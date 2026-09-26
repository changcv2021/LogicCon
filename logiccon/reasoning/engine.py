"""Four-module inference and the cumulative ablations in Appendix A.2."""
import time
from dataclasses import dataclass
from . import prompts
from .schema import parse_json, parse_claims, parse_verification, parse_diagnosis, require

MODES = {"base", "parsing", "verification", "aggregation", "full"}


@dataclass(frozen=True)
class MethodConfig:
    mode: str = "full"
    max_iterations: int = 3  # TOTAL verification rounds, including the first (§A.8).
    max_claims: int = 16
    format_retries: int = 1

    def __post_init__(self):
        require(self.mode in MODES, "Unknown ablation mode")
        require(type(self.max_iterations) is int and 1 <= self.max_iterations <= 3, "K must be 1..3")
        require(type(self.max_claims) is int and 1 <= self.max_claims <= 64, "max_claims must be 1..64")
        require(type(self.format_retries) is int and 0 <= self.format_retries <= 3, "format_retries must be 0..3")


class ModelOutputError(ValueError):
    pass


class Reasoner:
    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = config or MethodConfig()
        self.trace = []

    def call(self, stage, instruction, payload, image, validator):
        original_prompt = prompts.make_prompt(instruction, payload)
        prompt = original_prompt
        for attempt in range(self.config.format_retries + 1):
            started = time.monotonic()
            response = self.backend.generate(prompt, image=image)
            event = {"stage": stage, "attempt": attempt, "image_supplied": image is not None,
                     "prompt": prompt, "raw_response": response.text, "usage": response.usage,
                     "elapsed_seconds": time.monotonic() - started}
            self.trace.append(event)
            try:
                parsed = parse_json(response.text)
                result = validator(parsed)
                event["parsed"] = parsed
                return result
            except (ValueError, TypeError, KeyError) as exc:
                event["validation_error"] = str(exc)
                prompt = (original_prompt + "\nYour previous output failed validation: " + str(exc)
                          + "\nReturn a corrected JSON object following the original task. Previous output:\n"
                          + response.text)
        raise ModelOutputError(f"{stage} failed after {attempt + 1} attempts: {event['validation_error']}")

    def run(self, sample):
        self.trace = []
        data = {"statement": sample.statement}
        mode = self.config.mode
        claims, records, histories, compositional = [], [], {}, None
        if mode != "base":
            claims, compositional = self.call(
                "parse", prompts.PARSE, {**data, "max_claims": self.config.max_claims}, None,
                lambda row: parse_claims(row, self.config.max_claims))
            data["claims"] = [c.to_dict() for c in claims]

        if mode not in {"base", "parsing"}:
            for claim in claims:
                history = []
                for iteration in range(1, self.config.max_iterations + 1):
                    instruction = prompts.VERIFY + ("\n" + prompts.REFINE if history else "")
                    record = self.call(
                        "refine" if history else "verify", instruction,
                        {"statement": sample.statement, "claim": claim.to_dict(),
                         "previous_verifications": history, "iteration": iteration}, sample.image,
                        lambda row: parse_verification(row, claim, iteration))
                    history.append(record.to_dict())
                    if record.verdict != "insufficient":
                        break
                histories[claim.claim_id] = history
                records.append(record)
            data["verifications"] = [r.to_dict() for r in records]

        contradicted = {r.claim_id for r in records if r.verdict == "contradicted"}
        unresolved = {r.claim_id for r in records if r.verdict == "insufficient"}
        required_label = ("conflicting" if contradicted else
                          "consistent" if records and not unresolved else None)

        def grounded_diagnosis(row):
            # g is a model call (§5.2). These are protocol checks, not a rule-based g.
            allowed = contradicted or unresolved
            diagnosis = parse_diagnosis(row, allowed_ids=allowed, required_label=required_label)
            require(diagnosis["uncertain"] == bool(unresolved and not contradicted),
                    "uncertain must reflect unresolved evidence without an established contradiction")
            if diagnosis["label"] == "conflicting":
                cid = diagnosis["primary_claim_id"]
                claim = next(c for c in claims if c.claim_id == cid)
                record = next(r for r in records if r.claim_id == cid)
                require(diagnosis["target"] == record.target, "Copy verified target exactly")
                require(diagnosis["text_claim"] == claim.text_claim, "Copy original text_claim exactly")
                require(diagnosis["visual_evidence"] == record.visual_evidence, "Copy verified visual_evidence exactly")
            return diagnosis

        if mode in {"base", "parsing", "verification"}:
            # Cumulative ablation: a direct task answer sees only intermediates enabled so far.
            diagnosis = self.call("direct", prompts.DIRECT + "\n" + prompts.TAXONOMY + "\n" + prompts.DIAGNOSIS,
                                  data, sample.image, parse_diagnosis)
        else:
            diagnosis = self.call("aggregate", prompts.AGGREGATE + "\n" + prompts.TAXONOMY + "\n" + prompts.DIAGNOSIS,
                                  data, sample.image, grounded_diagnosis)
            if mode == "full":
                diagnosis = self.call("final", prompts.FINAL + "\n" + prompts.DIAGNOSIS,
                                      {**data, "diagnosis": diagnosis}, sample.image,
                                      lambda row: parse_diagnosis(row, previous=diagnosis))
        status = ("contradicted" if contradicted else "insufficient" if unresolved else
                  "supported" if records else "not_verified")
        points = [{"claim_id": c.claim_id, "target": r.target, "text_claim": c.text_claim,
                   "visual_evidence": r.visual_evidence}
                  for c, r in zip(claims, records) if r.verdict == "contradicted"]
        return {"sample_id": sample.sample_id, **diagnosis, "method": mode,
                "evidence_status": status, "predicted_compositional": compositional,
                "conflict_points": points, "claims": [c.to_dict() for c in claims],
                "verification_history": histories, "model_calls": len(self.trace),
                "usage": {k: sum(e["usage"].get(k, 0) for e in self.trace)
                          for k in ("input_tokens", "output_tokens")}}
