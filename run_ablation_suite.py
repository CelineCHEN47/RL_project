#!/usr/bin/env python3
"""Batch ablation runner for Tag RL experiments.

Runs a list of parameter variants across multiple random seeds, then writes
an aggregated ranking report so effective parameter settings can be compared.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AblationVariant:
    name: str
    env_overrides: dict[str, str]


DEFAULT_VARIANTS: list[AblationVariant] = [
    AblationVariant("baseline", {}),
    AblationVariant("tag_radius_small", {"TAG_RADIUS": "16"}),
    AblationVariant("tag_radius_large", {"TAG_RADIUS": "32"}),
    AblationVariant("cooldown_short", {"TAG_COOLDOWN_FRAMES": "5"}),
    AblationVariant("cooldown_long", {"TAG_COOLDOWN_FRAMES": "20"}),
    AblationVariant("prox_radius_small", {"PROXIMITY_RADIUS": "70"}),
    AblationVariant("prox_radius_large", {"PROXIMITY_RADIUS": "140"}),
    AblationVariant("prox_agg_mean", {"PROXIMITY_AGG": "mean"}),
    AblationVariant("shaping_weak", {"PROXIMITY_COEF": "0.08"}),
    AblationVariant("shaping_strong", {"PROXIMITY_COEF": "0.25"}),
]


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_best_eval(metrics: list[dict]) -> dict | None:
    if not metrics:
        return None
    for item in metrics:
        if item.get("epoch") == "best":
            return item
    return max(metrics, key=lambda m: float(m.get("mean_tags", 0.0)))


def _select_best_role(role_metrics: list[dict], role: str) -> dict | None:
    if not role_metrics:
        return None
    for item in role_metrics:
        if item.get("epoch") == "best":
            return item.get(role)
    return role_metrics[-1].get(role)


def _collect_run_metrics(run_dir: Path) -> dict[str, float]:
    eval_path = run_dir / "results" / "eval_metrics.json"
    role_path = run_dir / "results" / "role_eval_metrics.json"

    eval_metrics = _read_json(eval_path) or []
    role_metrics = _read_json(role_path) or []

    best_eval = _select_best_eval(eval_metrics) or {}
    best_tagger = _select_best_role(role_metrics, "tagger") or {}
    best_runner = _select_best_role(role_metrics, "runner") or {}

    overall_mean_tags = float(best_eval.get("mean_tags", 0.0))
    tagger_mean_tags = float(best_tagger.get("mean_tags", 0.0))
    runner_mean_tags = float(best_runner.get("mean_tags", 0.0))

    # Higher is better for overall and tagger metrics; lower is better for
    # runner mean_tags, so we subtract it.
    composite = overall_mean_tags + tagger_mean_tags - runner_mean_tags

    return {
        "overall_mean_tags": overall_mean_tags,
        "tagger_mean_tags": tagger_mean_tags,
        "runner_mean_tags": runner_mean_tags,
        "composite_score": composite,
    }


def build_command(args, run_dir: Path, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--algorithm",
        args.algorithm,
        "--save_dir",
        str(run_dir),
        "--epochs",
        *[str(e) for e in args.epochs],
        "--sims",
        str(args.sims),
        "--steps-per-round",
        str(args.steps_per_round),
        "--eval_episodes",
        str(args.eval_episodes),
        "--eval-steps",
        str(args.eval_steps),
        "--train-seed",
        str(seed),
        "--eval-seed",
        str(seed),
        "--trace-decimation",
        str(args.trace_decimation),
    ]
    return cmd


def run_suite(args) -> tuple[list[dict], list[dict]]:
    variants = DEFAULT_VARIANTS[: args.max_variants] if args.max_variants else DEFAULT_VARIANTS
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict] = []

    for variant in variants:
        for seed in args.seeds:
            run_dir = base_dir / variant.name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.update(variant.env_overrides)

            cmd = build_command(args, run_dir, seed)
            print(f"\n[RUN] {variant.name} | seed={seed}")
            print("      " + " ".join(cmd))
            if variant.env_overrides:
                print(f"      env={variant.env_overrides}")

            if not args.dry_run:
                subprocess.run(cmd, check=True, env=env)

            metrics = _collect_run_metrics(run_dir)
            record = {
                "variant": variant.name,
                "seed": seed,
                "run_dir": str(run_dir),
                "env_overrides": variant.env_overrides,
                **metrics,
            }
            run_records.append(record)
            print(f"      composite={metrics['composite_score']:.3f}")

    # Aggregate by variant
    by_variant: dict[str, list[dict]] = {}
    for r in run_records:
        by_variant.setdefault(r["variant"], []).append(r)

    aggregate: list[dict] = []
    for variant_name, rows in by_variant.items():
        overall = [x["overall_mean_tags"] for x in rows]
        tagger = [x["tagger_mean_tags"] for x in rows]
        runner = [x["runner_mean_tags"] for x in rows]
        composite = [x["composite_score"] for x in rows]
        aggregate.append(
            {
                "variant": variant_name,
                "n_runs": len(rows),
                "overall_mean_tags_mean": _safe_mean(overall),
                "overall_mean_tags_std": _safe_std(overall),
                "tagger_mean_tags_mean": _safe_mean(tagger),
                "tagger_mean_tags_std": _safe_std(tagger),
                "runner_mean_tags_mean": _safe_mean(runner),
                "runner_mean_tags_std": _safe_std(runner),
                "composite_score_mean": _safe_mean(composite),
                "composite_score_std": _safe_std(composite),
            }
        )

    aggregate.sort(key=lambda x: x["composite_score_mean"], reverse=True)
    return run_records, aggregate


def write_reports(base_dir: Path, run_records: list[dict], aggregate: list[dict]):
    summary = {
        "run_records": run_records,
        "aggregate_ranking": aggregate,
    }
    summary_path = base_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ranking_path = base_dir / "ranking.txt"
    with ranking_path.open("w", encoding="utf-8") as f:
        f.write("Ablation Ranking (higher composite is better)\n")
        f.write("=" * 60 + "\n")
        for idx, row in enumerate(aggregate, start=1):
            f.write(
                f"{idx:>2}. {row['variant']:<20} "
                f"composite={row['composite_score_mean']:.3f}±{row['composite_score_std']:.3f} | "
                f"overall={row['overall_mean_tags_mean']:.3f} | "
                f"tagger={row['tagger_mean_tags_mean']:.3f} | "
                f"runner={row['runner_mean_tags_mean']:.3f}\n"
            )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved ranking: {ranking_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-seed ablation suite and aggregate results."
    )
    parser.add_argument("--algorithm", default="PPO")
    parser.add_argument(
        "--base-dir",
        default="experiments/ablation_suite",
        help="Base output directory for all ablation runs.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", nargs="+", type=int, default=[20])
    parser.add_argument("--sims", type=int, default=2)
    parser.add_argument("--steps-per-round", type=int, default=1200)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=1200)
    parser.add_argument("--trace-decimation", type=int, default=20)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    run_records, aggregate = run_suite(args)
    write_reports(base_dir, run_records, aggregate)


if __name__ == "__main__":
    main()
