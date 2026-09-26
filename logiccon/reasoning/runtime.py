"""Resumable, auditable runs. One committed JSONL event contains prediction AND trace."""
import contextlib
import fcntl
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from ..io import digest, file_hash, jsonl
from . import prompts
from .backends import create_backend
from .engine import Reasoner
from .schema import Sample, require


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def write_jsonl(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def source_fingerprint():
    root = Path(__file__).parent
    sources = {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}
    sources["../io.py"] = file_hash(root.parent / "io.py")
    return digest(sources)


def environment():
    result = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "transformers", "accelerate", "Pillow"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def model_fingerprint(config):
    if config.kind != "transformers":
        return {"model": config.model, "revision": config.revision,
                "note": "Remote endpoint must independently pin its served weights"}
    path = Path(config.model)
    if path.is_dir():
        # Small configs/tokenizer metadata are hashed. Multi-GB weights use size/mtime.
        # An immutable snapshot remains necessary for bit-level weight provenance.
        small = {p.name: file_hash(p) for p in sorted(path.glob("*.json")) if p.stat().st_size < 2_000_000}
        weights = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                   for pattern in ("*.safetensors", "pytorch_model*.bin") for p in sorted(path.glob(pattern))}
        require(bool(small) and bool(weights), "Local model directory lacks configuration/weights")
        return {"path": str(path.resolve()), "metadata_sha256": small, "weight_file_stats": weights}
    require(len(config.revision) == 40 and all(c in "0123456789abcdef" for c in config.revision.lower()),
            "Hub model IDs require a 40-character immutable commit revision")
    return {"model": config.model, "revision": config.revision}


@contextlib.contextmanager
def locked_run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Run directory is already in use; give each shard its own directory") from None
        yield output


def initialize(output, metadata, resume):
    path = output / "manifest.json"
    if path.exists():
        require(resume, "Run already exists; use --resume or a new output directory")
        require(json.loads(path.read_text()) == metadata,
                "Resume refused: inputs/config/code/model/environment or shard selection changed")
    else:
        require(not (output / "events.jsonl").exists(), "Orphaned event log without manifest")
        atomic_json(path, metadata)


def read_events(path):
    """Recover only an incomplete final append; save it for inspection before truncation."""
    path = Path(path)
    records = {}
    if not path.exists():
        return records
    with path.open("rb+") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                backup = path.with_name(f"{path.name}.partial-{position}")
                if not backup.exists():
                    backup.write_bytes(line)
                stream.truncate(position)
                break
            row = json.loads(line)
            sid = row["sample_id"]
            require(records.get(sid, {}).get("status") != "ok", "Duplicate completed sample in event log")
            require(row.get("status") in {"ok", "error"}, "Invalid event status")
            records[sid] = row
    return records


def append_event(stream, event):
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def export_predictions(output, events):
    successful = [r["prediction"] for r in events.values() if r["status"] == "ok"]
    write_jsonl(output / "predictions.jsonl", sorted(successful, key=lambda r: r["sample_id"]))
    return successful


def selected_samples(inputs, image_root, limit, shard_index, num_shards):
    require(num_shards > 0 and 0 <= shard_index < num_shards, "Invalid shard index/count")
    require(limit is None or limit > 0, "limit must be positive")
    seen, samples = set(), []
    for index, row in enumerate(jsonl(inputs)):
        if limit is not None and index >= limit:
            break
        # All selected input rows are schema checked, including the other shards.
        require(isinstance(row, dict) and set(row) == {"sample_id", "image", "statement"},
                "Public input has forbidden or missing fields")
        sid = row["sample_id"]
        require(isinstance(sid, str) and sid not in seen, "Duplicate/invalid sample_id")
        seen.add(sid)
        if index % num_shards == shard_index:
            samples.append(Sample.from_dict(row, image_root))
    require(bool(seen), "Input dataset is empty")
    return samples


def run_inference(config, inputs, image_root, output, *, resume=False, limit=None,
                  shard_index=0, num_shards=1, keep_going=False, backend_factory=create_backend,
                  fingerprint=True):
    # The CLI enforces the login-node guard before this function is called.
    # Dependency injection is for tiny software tests, never a claimed model benchmark.
    samples = selected_samples(inputs, image_root, limit, shard_index, num_shards)
    manifest = {"schema": "logiccon-inference-1", "config": config.to_dict(),
                "inputs_sha256": file_hash(inputs), "image_root": str(Path(image_root).resolve()),
                "limit": limit, "shard_index": shard_index, "num_shards": num_shards,
                "source_sha256": source_fingerprint(), "prompt_version": prompts.VERSION,
                "environment": environment(),
                "model": model_fingerprint(config.backend) if fingerprint else {"software_test_only": True}}
    with locked_run(output) as root:
        initialize(root, manifest, resume)
        events = read_events(root / "events.jsonl")
        selected_ids = {s.sample_id for s in samples}
        require(not events.keys() - selected_ids, "Event log contains unknown sample IDs")
        reasoner = None
        try:
            with (root / "events.jsonl").open("a", encoding="utf-8") as log:
                for sample in samples:
                    sample_hash = digest([asdict(sample), file_hash(sample.image)])
                    previous = events.get(sample.sample_id)
                    if previous:
                        require(previous["sample_sha256"] == sample_hash, "Resume refused: image/sample changed")
                        if previous["status"] == "ok":
                            continue
                    if reasoner is None:
                        reasoner = Reasoner(backend_factory(config.backend), config.method)
                    try:
                        prediction = reasoner.run(sample)
                        event = {"status": "ok", "sample_id": sample.sample_id,
                                 "sample_sha256": sample_hash, "prediction": prediction,
                                 "trace": reasoner.trace}
                    except Exception as exc:
                        event = {"status": "error", "sample_id": sample.sample_id,
                                 "sample_sha256": sample_hash, "error_type": type(exc).__name__,
                                 "error": str(exc), "trace": reasoner.trace}
                        append_event(log, event)
                        events[sample.sample_id] = event
                        if not keep_going:
                            raise
                        continue
                    append_event(log, event)
                    events[sample.sample_id] = event
                    print(json.dumps({"sample_id": sample.sample_id, "status": "ok",
                                      "model_calls": prediction["model_calls"]}), flush=True)
        finally:
            successful = export_predictions(root, events)
            total_calls = sum(len(event.get("trace", [])) for event in jsonl(root / "events.jsonl"))
            atomic_json(root / "summary.json", {"selected": len(samples), "completed": len(successful),
                        "failed": sum(e["status"] == "error" for e in events.values()),
                        "missing": len(samples) - len(successful),
                        "model_calls": total_calls,
                        "uncertain": sum(p["uncertain"] for p in successful)})
        return json.loads((root / "summary.json").read_text())


def merge_runs(directories, output):
    rows, seen, manifests = [], set(), []
    for directory in directories:
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        require(manifest["schema"] == "logiccon-inference-1", "Expected inference run")
        manifests.append(manifest)
        summary = json.loads((root / "summary.json").read_text())
        require(summary["missing"] == 0, f"Shard incomplete: {root}")
        shard_rows = list(jsonl(root / "predictions.jsonl"))
        require(len(shard_rows) == summary["completed"], "Prediction count mismatch")
        for row in shard_rows:
            require(row["sample_id"] not in seen, "Duplicate sample_id across shards")
            seen.add(row["sample_id"])
            rows.append(row)
    require(bool(manifests), "No runs provided")
    common = [{k: v for k, v in m.items() if k != "shard_index"} for m in manifests]
    require(all(m == common[0] for m in common), "Shards have different configurations/data/code")
    require(sorted(m["shard_index"] for m in manifests) == list(range(manifests[0]["num_shards"])),
            "Provide every shard exactly once")
    require(not Path(output).exists(), "Refusing to overwrite merged predictions")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, sorted(rows, key=lambda r: r["sample_id"]))
    return {"samples": len(rows), "shards": len(manifests)}
