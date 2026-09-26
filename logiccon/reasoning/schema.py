"""Strict boundaries between public inputs, model outputs and private annotations."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

TYPES = {"attribute", "object", "relation", "spatial", "quantity"}
VERDICTS = {"supported", "contradicted", "insufficient"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")
    return value.strip()


def string_list(value, name):
    require(isinstance(value, list), f"{name} must be a list")
    return [nonempty(v, name) for v in value]


def parse_json(raw):
    """Accept JSON or a single fenced JSON object; never scrape arbitrary prose."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        require(lines[0] in {"```", "```json"} and lines[-1] == "```", "Invalid JSON fence")
        text = "\n".join(lines[1:-1])

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(text, object_pairs_hook=unique_pairs)
    require(isinstance(result, dict), "Model output must be a JSON object")
    return result


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: str
    statement: str

    @classmethod
    def from_dict(cls, row, image_root):
        require(isinstance(row, dict) and set(row) == {"sample_id", "image", "statement"},
                "Inference inputs must contain ONLY sample_id, image, statement; annotations are forbidden")
        for key, value in row.items():
            nonempty(value, key)
        root = Path(image_root).resolve()
        relative = Path(row["image"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Expected a relative image path")
        image = (root / relative).resolve()
        require(image.is_relative_to(root), "Image escapes image_root")
        require(image.is_file(), f"Image missing: {image}")
        require(image.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}, "Unsupported image suffix")
        return cls(row["sample_id"], str(image), row["statement"])


@dataclass(frozen=True)
class Claim:
    claim_id: str
    target: str
    predicate: str
    detail: str
    text_claim: str
    context: list[str]

    def to_dict(self):
        return asdict(self)


def parse_claims(row, max_claims):
    require(isinstance(row.get("claims"), list) and 0 < len(row["claims"]) <= max_claims,
            f"Expected 1..{max_claims} complete claims; do not truncate the statement")
    require(type(row.get("compositional")) is bool, "compositional must be boolean")
    claims, ids = [], set()
    for item in row["claims"]:
        require(isinstance(item, dict), "Claim must be an object")
        values = {key: nonempty(item.get(key), key) for key in
                  ("claim_id", "target", "predicate", "detail", "text_claim")}
        require(values["claim_id"] not in ids, "Duplicate claim_id")
        ids.add(values["claim_id"])
        claims.append(Claim(**values, context=string_list(item.get("context"), "context")))
    return claims, row["compositional"]


@dataclass(frozen=True)
class Verification:
    claim_id: str
    target: str
    visual_evidence: str
    verdict: str
    anchors: list[str]
    subclaims: list[str]
    missing_evidence: str
    iteration: int

    def to_dict(self):
        return asdict(self)


def parse_verification(row, claim, iteration):
    require(row.get("claim_id") == claim.claim_id, "Verification changed claim_id")
    require(isinstance(row.get("verdict"), str) and row["verdict"] in VERDICTS, "Invalid verdict")
    evidence = row.get("visual_evidence")
    require(isinstance(evidence, str), "visual_evidence must be a string")
    if row["verdict"] != "insufficient":
        nonempty(evidence, "Explicit visual evidence for a decisive verdict")
    missing = row.get("missing_evidence")
    require(isinstance(missing, str), "missing_evidence must be a string")
    if row["verdict"] == "insufficient":
        nonempty(missing, "Explain missing evidence")
    return Verification(claim.claim_id, nonempty(row.get("target"), "target"), evidence.strip(),
                        row["verdict"], string_list(row.get("anchors"), "anchors"),
                        string_list(row.get("subclaims"), "subclaims"), missing, iteration)


def parse_diagnosis(row, allowed_ids=None, required_label=None, previous=None):
    require(isinstance(row.get("label"), str) and row["label"] in {"consistent", "conflicting"}, "Invalid label")
    if required_label:
        require(row["label"] == required_label, f"Verified facts require label={required_label}")
    require(type(row.get("uncertain")) is bool, "uncertain must be boolean")
    nonempty(row.get("explanation"), "explanation")
    if row["label"] == "conflicting":
        require(isinstance(row.get("conflict_type"), str) and row["conflict_type"] in TYPES, "Invalid conflict type")
        for key in ("target", "text_claim", "visual_evidence"):
            nonempty(row.get(key), key)
        if allowed_ids is not None:
            require(row.get("primary_claim_id") in allowed_ids, "Select an eligible verified claim")
    else:
        require(row.get("conflict_type") is None, "Consistent output must have null conflict_type")
        require(row.get("primary_claim_id") is None, "Consistent output must have null primary_claim_id")
        for key in ("target", "text_claim", "visual_evidence"):
            require(row.get(key) == "", f"Consistent output must have empty {key}")
    if previous:
        for key in ("label", "conflict_type", "primary_claim_id", "target", "text_claim", "visual_evidence", "uncertain"):
            require(row.get(key) == previous.get(key), f"Final reasoning must preserve diagnosed {key}")
    return {key: row.get(key) for key in ("label", "conflict_type", "primary_claim_id", "target",
                                        "text_claim", "visual_evidence", "uncertain", "explanation")}
