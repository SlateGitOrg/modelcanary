"""Command-line interface: ``modelcanary <command>``.

Commands
--------
simulate   run one simulated rollout and print the decision and audit trail
evaluate   many-seed false-promotion / false-rollback evaluation of all strategies
tune       pilot (A/A) estimate of planning standard deviations
oc         exact and Wald operating characteristics of the error-rate SPRT
serve      start the FastAPI gateway over a directory of ONNX models
audit      verify an audit log's hash chain
demo       end-to-end demo with real scikit-learn -> ONNX models (needs the demo extra)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from modelcanary.audit import AuditLog, verify_file
from modelcanary.engine.monitors import Strategy
from modelcanary.engine.policy import RolloutPolicy
from modelcanary.errors import ModelCanaryError
from modelcanary.evaluation.analytic import render_oc_markdown, sprt_oc_table
from modelcanary.evaluation.harness import EVAL_SEED_START
from modelcanary.evaluation.runner import EvaluationConfig, run_evaluation
from modelcanary.evaluation.tuning import tune_policy
from modelcanary.logging_setup import configure_logging
from modelcanary.sim.runner import simulate_rollout
from modelcanary.sim.scenarios import BASELINE, standard_scenarios

DEFAULT_POLICY = os.environ.get("MODELCANARY_POLICY", "configs/default.toml")


def _policy(path: str) -> RolloutPolicy:
    return RolloutPolicy.from_file(path)


def _cmd_simulate(args: argparse.Namespace) -> int:
    scenarios = standard_scenarios()
    if args.scenario not in scenarios:
        print(
            f"unknown scenario {args.scenario!r}; choose from {sorted(scenarios)}", file=sys.stderr
        )
        return 2
    audit = AuditLog(args.audit) if args.audit else AuditLog()
    result = simulate_rollout(
        scenarios[args.scenario],
        _policy(args.policy),
        args.seed,
        Strategy(args.strategy),
        audit=audit,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    sc = scenarios[args.scenario]
    print(f"scenario  : {sc.name} -- {sc.description} (truth: {result.truth.value})")
    print(f"strategy  : {result.strategy.value}   seed: {result.seed}")
    print(
        f"decision  : {result.action.value.upper()} at stage {result.stage!r} after "
        f"{result.requests:,} requests ({result.candidate_requests:,} served by the candidate)"
    )
    print(f"reason    : {result.reason}")
    print("audit trail:")
    for rec in audit.recent(10_000):
        p = rec.payload
        if rec.event == "verdict_changed":
            detail = (
                f"{p['metric']}: {p['from']} -> {p['verdict']} at {p['requests']:,} requests "
                f"({p['pairs']:,} pairs)"
            )
        elif rec.event == "stage_advanced":
            detail = f"{p['from']} -> {p['to']} at {p['requests']:,} requests"
        elif rec.event == "rollout_started":
            detail = f"{p['baseline']} vs {p['candidate']} ({p['strategy']})"
        else:
            detail = p.get("reason", "")
        print(f"  #{rec.seq:<3} {rec.event:<16} {detail}")
    print(f"audit chain verified: {audit.verify().ok}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = EvaluationConfig(
        seeds=args.seeds,
        pilot_seeds=args.pilot_seeds,
        workers=args.workers,
        batch_size=args.batch_size,
        scenarios=tuple(args.scenario or ()),
    )
    report = run_evaluation(_policy(args.policy), config)
    print(report.markdown)
    if args.out:
        md, js = report.write(args.out)
        print(f"wrote {md} and {js}")
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    tuned, sds = tune_policy(_policy(args.policy), BASELINE, list(range(args.pilot_seeds)))
    for name, sd in sds.items():
        print(f"{name:<14} planning_sd = {sd:.6f}")
    if args.out:
        Path(args.out).write_text(tuned.to_json(), encoding="utf-8")
        print(f"wrote tuned policy to {args.out}")
    return 0


def _cmd_oc(args: argparse.Namespace) -> int:
    p1, alpha, beta, rows = sprt_oc_table(_policy(args.policy))
    print(render_oc_markdown(p1, alpha, beta, rows))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from modelcanary.serving.app import create_app
    from modelcanary.serving.feedback import FeedbackStore
    from modelcanary.serving.gateway import Gateway
    from modelcanary.serving.registry import ModelRegistry

    registry = ModelRegistry()
    registry.load_directory(args.models)
    gateway = Gateway(
        registry,
        args.production,
        default_policy=_policy(args.policy),
        audit=AuditLog(args.audit) if args.audit else AuditLog(),
        feedback=FeedbackStore(ttl_seconds=args.feedback_ttl),
    )
    uvicorn.run(create_app(gateway), host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    result = verify_file(args.path)
    if result.ok:
        print(f"OK: {result.records} records, hash chain intact")
        return 0
    print(f"BROKEN at seq {result.first_bad_seq}: {result.reason}")
    return 1


def _cmd_demo(args: argparse.Namespace) -> int:
    try:
        from modelcanary.demo.run import run_demo
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        print(f"the demo needs the 'demo' extra (scikit-learn, skl2onnx, httpx2): {exc}")
        return 2
    return run_demo(Path(args.out), seed=args.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelcanary", description=__doc__.splitlines()[0])
    parser.add_argument("--log-level", default=os.environ.get("MODELCANARY_LOG_LEVEL", "WARNING"))
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run one simulated rollout")
    sim.add_argument("--scenario", default="worse-error")
    sim.add_argument("--seed", type=int, default=EVAL_SEED_START)
    sim.add_argument("--strategy", choices=[s.value for s in Strategy], default="sequential")
    sim.add_argument("--policy", default=DEFAULT_POLICY)
    sim.add_argument("--audit", help="append the audit trail to this JSONL file")
    sim.add_argument("--json", action="store_true")
    sim.set_defaults(func=_cmd_simulate)

    ev = sub.add_parser("evaluate", help="many-seed evaluation of all strategies")
    ev.add_argument("--seeds", type=int, default=300)
    ev.add_argument("--pilot-seeds", type=int, default=20)
    ev.add_argument("--workers", type=int, default=1)
    ev.add_argument("--batch-size", type=int, default=500)
    ev.add_argument("--scenario", action="append", help="restrict to a scenario (repeatable)")
    ev.add_argument("--policy", default=DEFAULT_POLICY)
    ev.add_argument("--out", help="directory for eval.md and eval.json")
    ev.set_defaults(func=_cmd_evaluate)

    tune = sub.add_parser("tune", help="pilot estimate of planning standard deviations")
    tune.add_argument("--pilot-seeds", type=int, default=20)
    tune.add_argument("--policy", default=DEFAULT_POLICY)
    tune.add_argument("--out", help="write the tuned policy as JSON")
    tune.set_defaults(func=_cmd_tune)

    oc = sub.add_parser("oc", help="operating characteristics of the error-rate SPRT")
    oc.add_argument("--policy", default=DEFAULT_POLICY)
    oc.set_defaults(func=_cmd_oc)

    serve = sub.add_parser("serve", help="run the HTTP gateway")
    serve.add_argument("--models", default=os.environ.get("MODELCANARY_MODELS_DIR", "models"))
    serve.add_argument(
        "--production", default=os.environ.get("MODELCANARY_PRODUCTION"), required=False
    )
    serve.add_argument("--policy", default=DEFAULT_POLICY)
    serve.add_argument("--audit", default=os.environ.get("MODELCANARY_AUDIT_PATH"))
    serve.add_argument("--host", default=os.environ.get("MODELCANARY_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("MODELCANARY_PORT", "8000")))
    serve.add_argument("--feedback-ttl", type=float, default=3600.0)
    serve.set_defaults(func=_cmd_serve)

    audit = sub.add_parser("audit", help="verify an audit log")
    audit.add_argument("path")
    audit.set_defaults(func=_cmd_audit)

    demo = sub.add_parser("demo", help="end-to-end demo with real ONNX models")
    demo.add_argument("--out", default="var/demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    if args.command == "serve" and not args.production:
        parser.error("serve needs --production (or MODELCANARY_PRODUCTION)")
    try:
        code: int = args.func(args)
    except ModelCanaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code
