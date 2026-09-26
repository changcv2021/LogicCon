"""Command-line entry points; inference and bulk work are blocked on login nodes."""
import argparse
import json
import platform
import socket
from ..io import compute_guard, write_json
from .config import load_config
from .engine import MODES
from .runtime import environment, merge_runs, run_inference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-config", help="Validate settings without loading a model")
    validate.add_argument("--config", required=True)
    doctor = sub.add_parser("doctor", help="Lightweight dependency discovery; does not import torch")
    doctor.add_argument("--config")
    infer = sub.add_parser("infer", help="Run §5 method or a cumulative ablation")
    infer.add_argument("--config", required=True)
    infer.add_argument("--inputs", required=True)
    infer.add_argument("--image-root", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--mode", choices=sorted(MODES))
    infer.add_argument("--limit", type=int)
    infer.add_argument("--shard-index", type=int, default=0)
    infer.add_argument("--num-shards", type=int, default=1)
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--keep-going", action="store_true")
    judge = sub.add_parser("judge", help="Isolated semantic judging using independent gold records")
    judge.add_argument("--config", required=True)
    judge.add_argument("--gold", required=True)
    judge.add_argument("--predictions", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--judge-id", required=True)
    judge.add_argument("--resume", action="store_true")
    score = sub.add_parser("score", help="Compute §A.7 metrics")
    score.add_argument("--gold", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--judgments")
    score.add_argument("--output", required=True)
    merge = sub.add_parser("merge", help="Merge every completed inference shard")
    merge.add_argument("--runs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            result = load_config(args.config).to_dict()
        elif args.command == "doctor":
            result = {"hostname": socket.gethostname(), "python": platform.python_version(),
                      "environment": environment(), "gpu_execution_tested": False}
            if args.config:
                result["config"] = load_config(args.config).to_dict()
        else:
            compute_guard()
            if args.command == "infer":
                result = run_inference(load_config(args.config, args.mode), args.inputs, args.image_root,
                                       args.output, resume=args.resume, limit=args.limit,
                                       shard_index=args.shard_index, num_shards=args.num_shards,
                                       keep_going=args.keep_going)
            elif args.command == "judge":
                from .judge import run_judge
                result = run_judge(load_config(args.config), args.gold, args.predictions,
                                   args.output, args.judge_id, resume=args.resume)
            elif args.command == "score":
                from .metrics import score as calculate
                result = calculate(args.gold, args.predictions, args.judgments)
                write_json(args.output, result)
            else:
                result = merge_runs(args.runs, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "infer" and result["missing"]:
            return 1
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
