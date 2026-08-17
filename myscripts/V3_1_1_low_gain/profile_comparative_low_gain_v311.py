"""Run the balanced isolated Baseline/CA/conservative-Refine profile."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from myscripts.V3_1_1 import profile_comparative_v311 as official
from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE
from myscripts.V3_1_1_low_gain.runtime import require_scale


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = official.build_parser()
    parser.description = __doc__
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    return parser


def _base_argv(args: argparse.Namespace) -> list[str]:
    return [
        "--baseline-weights", args.baseline_weights,
        "--ca-weights", args.ca_weights,
        "--refine-profile-summary", args.refine_profile_summary,
        "--data", args.data,
        "--split", args.split,
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", str(args.device),
        "--workers", str(args.workers),
        "--no-amp" if not args.amp else "--amp",
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--stability-relative-tolerance", str(args.stability_relative_tolerance),
        "--memory-relative-tolerance", str(args.memory_relative_tolerance),
        "--max-images", str(args.max_images),
        "--output-dir", args.output_dir,
    ]


def _low_gain_worker(scale: float, original_worker, **kwargs) -> dict[str, Any]:
    if kwargs["method"] != "CA+Refine":
        return original_worker(**kwargs)

    repeat = kwargs["repeat"]
    order_position = kwargs["order_position"]
    args = kwargs["args"]
    refine_summary = kwargs["refine_summary"]
    output_dir = kwargs["output_dir"]
    run_dir = output_dir / "isolated_runs" / f"repeat_{repeat:02d}_position_{order_position:02d}_ca_refine"
    command = [
        sys.executable,
        "-m",
        "myscripts.V3_1_1_low_gain.profile_refine_low_gain_v311",
        "--checkpoint", str(refine_summary["refine_checkpoint"]),
        "--ca-weights", args.ca_weights,
        "--data", args.data,
        "--split", args.split,
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", str(args.device),
        "--workers", str(args.workers),
        "--no-amp",
        "--warmup", str(args.warmup),
        "--max-images", str(args.max_images),
        "--residual-scale", str(scale),
        "--output-dir", str(run_dir),
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    print(
        f"[low-gain comparative] repeat={repeat} position={order_position} "
        f"method=CA+Refine isolated_output={run_dir}"
    )
    result = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"isolated low-gain profile failed: repeat={repeat}, position={order_position}, "
            f"returncode={result.returncode}"
        )
    summary_path = run_dir / "profile_summary.json"
    rows_path = run_dir / "profile_per_image.csv"
    if not summary_path.is_file() or not rows_path.is_file():
        raise RuntimeError(f"isolated low-gain worker did not create complete outputs: {run_dir}")
    summary = official._read_json(summary_path)
    if float(summary.get("residual_scale", -1.0)) != scale:
        raise RuntimeError("isolated low-gain worker used an unexpected residual scale")
    return {
        "method": "CA+Refine",
        "repeat": repeat,
        "order_position": order_position,
        "run_dir": str(run_dir),
        "summary": summary,
        "rows": official._read_csv(rows_path),
    }


def _annotate_outputs(output_dir: Path, residual_scale: float) -> None:
    audit_path = output_dir / "comparative_profile.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "tool": "profile_comparative_low_gain_v311",
            "residual_scale": residual_scale,
            "all_refine_workers_used_residual_scale": True,
        }
    )
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for filename in ("comparative_latency.csv", "comparative_repeat_summary.csv", "comparative_per_image.csv"):
        path = output_dir / filename
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["residual_scale"] = residual_scale if row.get("method") == "CA+Refine" else ""
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    official.validate_args(parser, args)
    require_scale(parser, args.residual_scale)
    reference = json.loads(Path(args.refine_profile_summary).read_text(encoding="utf-8"))
    if float(reference.get("residual_scale", -1.0)) != args.residual_scale:
        parser.error("refine-profile-summary residual scale does not match --residual-scale")

    original_worker = official._run_isolated_worker

    def patched_worker(**kwargs):
        return _low_gain_worker(args.residual_scale, original_worker, **kwargs)

    official._run_isolated_worker = patched_worker
    try:
        official.main(_base_argv(args))
    finally:
        official._run_isolated_worker = original_worker
    _annotate_outputs(Path(args.output_dir), args.residual_scale)


if __name__ == "__main__":
    main()
