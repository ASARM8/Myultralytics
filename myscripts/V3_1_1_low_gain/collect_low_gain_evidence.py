"""Collect a self-contained evidence package for the frozen low-gain setting."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from dataclasses import dataclass

from myscripts.V3_1_1.evidence_runtime import CANONICAL_BASELINE_WEIGHTS, CANONICAL_CA_WEIGHTS
from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DATA = "/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml"
CANONICAL_REFINE = (
    "/root/autodl-tmp/paper_exports/refine_v311_seed0/"
    "train_geometry_only/checkpoints/best.pt"
)
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    command: tuple[str, ...]
    expected: tuple[Path, ...] = ()


def _module(name: str, *args) -> tuple[str, ...]:
    return (sys.executable, "-m", name, *(str(value) for value in args))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=CANONICAL_DATA)
    parser.add_argument("--baseline-weights", default=CANONICAL_BASELINE_WEIGHTS)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--checkpoint", default=CANONICAL_REFINE)
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qualitative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--archive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=f"/root/autodl-tmp/paper_exports/ivc_low_gain_evidence_{timestamp}",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.workers < 0:
        parser.error("workers must be non-negative")
    if abs(args.residual_scale - DEFAULT_RESIDUAL_SCALE) > 1e-12:
        parser.error(f"formal low-gain collection is locked to residual-scale={DEFAULT_RESIDUAL_SCALE}")
    for label, value in (
        ("data", args.data),
        ("Baseline checkpoint", args.baseline_weights),
        ("CA checkpoint", args.ca_weights),
        ("Refine checkpoint", args.checkpoint),
    ):
        if not Path(value).exists():
            parser.error(f"{label} not found: {value}")


def build_stages(args: argparse.Namespace, output_dir: Path) -> list[Stage]:
    stages: list[Stage] = []
    if args.tests:
        stages.append(
            Stage(
                "tests",
                "运行低指标与正式 V3.1.1 协议测试",
                _module("pytest", "-q", "tests/V3_1_1_low_gain", "tests/V3_1_1"),
            )
        )

    main_csv = output_dir / "baseline_ca_validation.csv"
    stages.append(
        Stage(
            "baseline_ca_validation",
            "采集 Baseline 与 CA 完整 AP 阈值主结果",
            _module(
                "myscripts.paper_visuals.collect_validation_metrics",
                "--model", f"Baseline={args.baseline_weights}",
                "--model", f"CA={args.ca_weights}",
                "--data", args.data,
                "--imgsz", 640,
                "--batch", 8,
                "--device", args.device,
                "--workers", args.workers,
                "--project", output_dir / "baseline_ca_validation_runs",
                "--output-csv", main_csv,
            ),
            (main_csv,),
        )
    )

    validations = (
        ("reproduction_fp32_batch8", 8, "--no-amp"),
        ("reproduction_amp_batch8", 8, "--amp"),
        ("reproduction_fp32_batch1", 1, "--no-amp"),
    )
    for name, batch, amp_flag in validations:
        destination = output_dir / name
        stages.append(
            Stage(
                name,
                f"执行{name}低指标验证",
                _module(
                    "myscripts.V3_1_1_low_gain.validate_low_gain_v311",
                    "--checkpoint", args.checkpoint,
                    "--ca-weights", args.ca_weights,
                    "--data", args.data,
                    "--imgsz", 640,
                    "--batch", batch,
                    "--device", args.device,
                    "--workers", args.workers,
                    amp_flag,
                    "--residual-scale", args.residual_scale,
                    "--target-gain", 0.04,
                    "--output-dir", destination,
                ),
                (destination / "val_metrics.csv", destination / "low_gain_audit.json"),
            )
        )

    combined_csv = output_dir / "main_results_low_gain.csv"
    stages.append(
        Stage(
            "assemble_main_results",
            "合并 Baseline、CA 和低指标 Refine 主结果",
            _module(
                "myscripts.V3_1_1_low_gain.assemble_main_results_low_gain",
                "--baseline-ca-csv", main_csv,
                "--low-gain-metrics-csv", output_dir / "reproduction_fp32_batch8" / "val_metrics.csv",
                "--residual-scale", args.residual_scale,
                "--output-csv", combined_csv,
            ),
            (combined_csv, combined_csv.with_suffix(".audit.json")),
        )
    )

    audit_dir = output_dir / "reproduction_audit"
    stages.append(
        Stage(
            "reproduction_audit",
            "审计 FP32/AMP 与 batch 1/8 三条评估路径",
            _module(
                "myscripts.V3_1_1_low_gain.audit_reproductions_low_gain_v311",
                "--fp32-batch8-dir", output_dir / "reproduction_fp32_batch8",
                "--amp-batch8-dir", output_dir / "reproduction_amp_batch8",
                "--fp32-batch1-dir", output_dir / "reproduction_fp32_batch1",
                "--residual-scale", args.residual_scale,
                "--metric-tolerance", 0.001,
                "--output-dir", audit_dir,
            ),
            (audit_dir / "reproduction_audit.json", audit_dir / "reproduction_audit.md"),
        )
    )

    profile_dir = output_dir / "profile_refine_fp32_batch1"
    stages.append(
        Stage(
            "profile_refine_fp32_batch1",
            "测量低指标完整推理链性能",
            _module(
                "myscripts.V3_1_1_low_gain.profile_refine_low_gain_v311",
                "--checkpoint", args.checkpoint,
                "--ca-weights", args.ca_weights,
                "--data", args.data,
                "--split", "val",
                "--imgsz", 640,
                "--batch", 1,
                "--device", args.device,
                "--workers", args.workers,
                "--no-amp",
                "--warmup", 500,
                "--residual-scale", args.residual_scale,
                "--output-dir", profile_dir,
            ),
            (profile_dir / "profile_summary.json", profile_dir / "profile_per_image.csv"),
        )
    )

    comparative_dir = output_dir / "profile_comparative_fp32_batch1"
    stages.append(
        Stage(
            "profile_comparative_fp32_batch1",
            "三轮隔离比较 Baseline、CA 与低指标 Refine",
            _module(
                "myscripts.V3_1_1_low_gain.profile_comparative_low_gain_v311",
                "--baseline-weights", args.baseline_weights,
                "--ca-weights", args.ca_weights,
                "--refine-profile-summary", profile_dir / "profile_summary.json",
                "--data", args.data,
                "--split", "val",
                "--imgsz", 640,
                "--batch", 1,
                "--device", args.device,
                "--workers", args.workers,
                "--no-amp",
                "--warmup", 500,
                "--repeats", 3,
                "--residual-scale", args.residual_scale,
                "--output-dir", comparative_dir,
            ),
            (comparative_dir / "comparative_latency.csv", comparative_dir / "comparative_profile.json"),
        )
    )

    if args.qualitative:
        qualitative_dir = output_dir / "qualitative_predictions"
        stages.append(
            Stage(
                "qualitative_predictions",
                "导出低指标 GT/Baseline/CA/Refine 完整定性结果",
                _module(
                    "myscripts.V3_1_1_low_gain.export_qualitative_low_gain_v311",
                    "--baseline-weights", args.baseline_weights,
                    "--ca-weights", args.ca_weights,
                    "--checkpoint", args.checkpoint,
                    "--data", args.data,
                    "--split", "val",
                    "--imgsz", 640,
                    "--batch", 8,
                    "--device", args.device,
                    "--workers", args.workers,
                    "--no-amp",
                    "--copy-images",
                    "--exist-ok",
                    "--residual-scale", args.residual_scale,
                    "--output-dir", qualitative_dir,
                ),
                (qualitative_dir / "manifest_all.csv", qualitative_dir / "export_audit.json"),
            )
        )
        figure_dir = output_dir / "qualitative_figure"
        stages.append(
            Stage(
                "qualitative_figure",
                "按固定规则生成改善、一般和失败案例面板",
                _module(
                    "myscripts.paper_submission.prepare_ivc_qualitative",
                    "--manifest", qualitative_dir / "manifest_all.csv",
                    "--output-dir", figure_dir,
                ),
                (figure_dir / "fig6_qualitative_manifest.csv", figure_dir / "fig6_ivc_qualitative.png"),
            )
        )
    return stages


def _load_state(path: Path) -> dict:
    if not path.is_file():
        return {"protocol_version": PROTOCOL_VERSION, "stages": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("existing output directory uses a different low-gain protocol version")
    return state


def _write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_stage(stage: Stage, output_dir: Path, state_path: Path, state: dict, *, resume: bool) -> None:
    record = state["stages"].get(stage.name, {})
    if resume and record.get("status") == "completed" and all(path.exists() for path in stage.expected):
        print(f"[resume] {stage.name}: outputs already complete")
        return
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(stage.command)
    (logs / f"{stage.name}.command.txt").write_text(command_text + "\n", encoding="utf-8")
    print(f"\n=== {stage.title} ===")
    print(command_text)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["MPLCONFIGDIR"] = str(output_dir / ".mplconfig")
    state["stages"][stage.name] = {"status": "running", "command": command_text}
    _write_state(state_path, state)
    with (logs / f"{stage.name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            stage.command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()
    if returncode != 0:
        state["stages"][stage.name] = {"status": "failed", "returncode": returncode, "command": command_text}
        _write_state(state_path, state)
        raise RuntimeError(f"stage failed: {stage.name}, returncode={returncode}")
    missing = [str(path) for path in stage.expected if not path.exists()]
    if missing:
        state["stages"][stage.name] = {
            "status": "failed",
            "returncode": 0,
            "command": command_text,
            "missing_outputs": missing,
        }
        _write_state(state_path, state)
        raise RuntimeError(f"stage {stage.name} did not create expected outputs: {missing}")
    state["stages"][stage.name] = {"status": "completed", "returncode": 0, "command": command_text}
    _write_state(state_path, state)


def _finalize(args: argparse.Namespace, output_dir: Path, state: dict) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": commit,
        "data": args.data,
        "baseline_weights": args.baseline_weights,
        "ca_weights": args.ca_weights,
        "refine_checkpoint": args.checkpoint,
        "residual_scale": args.residual_scale,
        "imgsz": 640,
        "split": "val",
        "test_used": False,
        "weights_modified": False,
        "stages": state["stages"],
        "mechanism_evidence": "Reuse the frozen Baseline-to-CA H1/H2 evidence; Refine residual scaling does not alter assignment.",
    }
    (output_dir / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "evidence_sha256.txt":
            continue
        lines.append(f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "evidence_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.archive:
        archive = output_dir.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(output_dir, arcname=output_dir.name)
        print(archive)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if not args.allow_dirty:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        if dirty:
            parser.error("Git working tree is dirty; commit changes or pass --allow-dirty for a diagnostic run")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Low-gain evidence output: {output_dir}")
    state_path = output_dir / "run_state.json"
    state = _load_state(state_path)
    identity = {
        "data": args.data,
        "baseline_weights": args.baseline_weights,
        "ca_weights": args.ca_weights,
        "checkpoint": args.checkpoint,
        "residual_scale": args.residual_scale,
        "device": args.device,
        "workers": args.workers,
    }
    if "identity" in state and state["identity"] != identity:
        raise RuntimeError("resume identity changed; use a new output directory")
    state["identity"] = identity
    _write_state(state_path, state)
    for stage in build_stages(args, output_dir):
        _run_stage(stage, output_dir, state_path, state, resume=args.resume)
    _finalize(args, output_dir, state)
    print(output_dir)


if __name__ == "__main__":
    main()
