"""Orchestrate the complete locked IVC evidence collection workflow.

This entry point calls the existing, individually audited collectors.  It does
not train Baseline, CA, or Refine models.  H1/H2 uses a frozen, single-split,
assigner-only pass with zero optimization steps.  Latency and peak allocated
memory are additionally measured for Baseline, CA, and CA+Refine through three
balanced rounds of isolated synchronized workers.

The workflow is intentionally fixed to the validation split, ``imgsz=640``,
the canonical Baseline/CA checkpoints, and the selected Refine checkpoint.
Every subprocess receives a separate log and is tracked in ``run_state.json``
so an interrupted run can be continued with ``--resume``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from myscripts.V3_1_1.evidence_runtime import CANONICAL_BASELINE_WEIGHTS, CANONICAL_CA_WEIGHTS
from myscripts.V3_1_1.profile_refine_v311 import OFFICIAL_WARMUP_PASSES, PROFILE_PROTOCOL_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DATA = "/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml"
CANONICAL_REFINE = (
    "/root/autodl-tmp/paper_exports/refine_v311_seed0/"
    "train_geometry_only/checkpoints/best.pt"
)
CANONICAL_TRAINING_DIR = "/root/autodl-tmp/paper_exports/refine_v311_seed0/train_geometry_only"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/paper_exports"
EVIDENCE_PROTOCOL_VERSION = PROFILE_PROTOCOL_VERSION


@dataclass(frozen=True)
class CommandStage:
    """One auditable subprocess stage."""

    name: str
    title: str
    command: tuple[str, ...]
    expected: tuple[Path, ...] = ()
    allow_audit_threshold_failure: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键采集 IVC 投稿版所需的完整验证、诊断、效率和定性证据。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default=CANONICAL_DATA)
    parser.add_argument("--baseline-weights", default=CANONICAL_BASELINE_WEIGHTS)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--checkpoint", default=CANONICAL_REFINE)
    parser.add_argument("--training-dir", default=CANONICAL_TRAINING_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="省略时在 /root/autodl-tmp/paper_exports 下创建带时间戳的新目录。",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--h1h2-passes", type=int, default=1)
    parser.add_argument(
        "--tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="运行 V3.1.1 的无训练协议测试。",
    )
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="正式 profile/定性导出前分别试跑 20/16 张图。",
    )
    parser.add_argument(
        "--h1h2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="运行 Baseline 与 CA 各自的固定权重、单split H1/H2只读统计。",
    )
    parser.add_argument(
        "--archive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="完成后在输出目录旁生成 tar.gz。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="继续已有 --output-dir；已完成且输出仍完整的阶段会被跳过。",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if Path(args.baseline_weights).as_posix() != Path(CANONICAL_BASELINE_WEIGHTS).as_posix():
        parser.error(f"Baseline checkpoint 固定为: {CANONICAL_BASELINE_WEIGHTS}")
    if Path(args.ca_weights).as_posix() != Path(CANONICAL_CA_WEIGHTS).as_posix():
        parser.error(f"CA checkpoint 固定为: {CANONICAL_CA_WEIGHTS}")
    if args.workers < 0:
        parser.error("--workers 不能小于 0")
    if args.h1h2_passes < 1:
        parser.error("--h1h2-passes 必须大于等于 1")
    if args.resume and args.output_dir is None:
        parser.error("--resume 必须同时指定已有的 --output-dir")


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def expected_are_complete(paths: Iterable[Path]) -> bool:
    return all(file_is_complete(path) for path in paths)


def build_run_identity(args: argparse.Namespace) -> dict[str, object]:
    """Fields that must not change when an interrupted evidence run resumes."""
    git_code, git_commit = capture(["git", "rev-parse", "HEAD"])
    if git_code != 0 or not git_commit:
        raise RuntimeError("无法确定当前 Git commit，拒绝创建不可追溯的正式证据包")
    return {
        "evidence_protocol_version": EVIDENCE_PROTOCOL_VERSION,
        "git_commit": git_commit,
        "data": str(Path(args.data)),
        "baseline_weights": str(Path(args.baseline_weights)),
        "ca_weights": str(Path(args.ca_weights)),
        "refine_checkpoint": str(Path(args.checkpoint)),
        "training_dir": str(Path(args.training_dir)),
        "imgsz": 640,
        "split": "val",
        "device": str(args.device),
        "workers": int(args.workers),
        "h1h2_passes": int(args.h1h2_passes),
    }


def display_command(command: Iterable[str]) -> str:
    return shlex.join(str(item) for item in command)


class EvidenceCollector:
    """Run, log, validate, and resume the evidence stages."""

    def __init__(self, output_dir: Path, *, resume: bool) -> None:
        self.output_dir = output_dir
        self.logs_dir = output_dir / "logs"
        self.state_path = output_dir / "run_state.json"
        self.resume = resume
        self.environment = os.environ.copy()
        self.environment["OMP_NUM_THREADS"] = "1"
        self.environment["PYTHONHASHSEED"] = "0"
        if self.state_path.exists():
            self.state = read_json(self.state_path)
        else:
            self.state = {
                "tool": "collect_ivc_evidence",
                "created_at": iso_now(),
                "output_dir": str(output_dir),
                "protocol": {
                    "version": EVIDENCE_PROTOCOL_VERSION,
                    "split": "val",
                    "test_used": False,
                    "imgsz": 640,
                    "official_validation": "FP32, batch=8",
                    "official_latency": (
                        "FP32, batch=1, 500 warmup passes, 3x3 Latin-square order, "
                        "isolated synchronized workers"
                    ),
                },
                "stages": {},
            }

    def save_state(self) -> None:
        self.state["updated_at"] = iso_now()
        write_json(self.state_path, self.state)

    def lock_run_identity(self, identity: dict[str, object]) -> None:
        existing = self.state.get("run_identity")
        if existing is not None and existing != identity:
            differences = {
                key: {"existing": existing.get(key), "requested": identity.get(key)}
                for key in sorted(set(existing) | set(identity))
                if existing.get(key) != identity.get(key)
            }
            raise RuntimeError(
                "--resume 的关键协议或输入与原运行不一致，拒绝混合证据: "
                + json.dumps(differences, ensure_ascii=False)
            )
        self.state["run_identity"] = identity
        self.save_state()

    def _can_skip(self, name: str, expected: tuple[Path, ...]) -> bool:
        if not self.resume:
            return False
        status = self.state.get("stages", {}).get(name, {}).get("status")
        if status not in {"completed", "threshold_not_met"} or not expected_are_complete(expected):
            return False
        if name == "profile_refine_fp32_batch1":
            summary_path = self.output_dir / "profile_refine_fp32_batch1" / "profile_summary.json"
            summary = read_json(summary_path)
            return (
                summary.get("protocol_version") == EVIDENCE_PROTOCOL_VERSION
                and summary.get("warmup_passes") == OFFICIAL_WARMUP_PASSES
            )
        if name == "profile_comparative_fp32_batch1":
            summary_path = self.output_dir / "profile_comparative_fp32_batch1" / "comparative_profile.json"
            summary = read_json(summary_path)
            return (
                summary.get("protocol_version") == EVIDENCE_PROTOCOL_VERSION
                and summary.get("warmup_passes") == OFFICIAL_WARMUP_PASSES
            )
        return True

    def run_action(
        self,
        name: str,
        title: str,
        action: Callable[[], None],
        expected: tuple[Path, ...],
    ) -> None:
        if self._can_skip(name, expected):
            print(f"[SKIP] {title}")
            return
        print(f"\n{'=' * 88}\n[{name}] {title}\n{'=' * 88}")
        record = {
            "title": title,
            "status": "running",
            "started_at": iso_now(),
            "expected": [str(path) for path in expected],
        }
        self.state["stages"][name] = record
        self.save_state()
        try:
            action()
            if not expected_are_complete(expected):
                missing = [str(path) for path in expected if not file_is_complete(path)]
                raise RuntimeError(f"阶段输出缺失或为空: {missing}")
        except Exception as error:
            record.update(status="failed", finished_at=iso_now(), error=repr(error))
            self.save_state()
            raise
        record.update(status="completed", finished_at=iso_now())
        self.save_state()

    def run_command(self, stage: CommandStage) -> None:
        if self._can_skip(stage.name, stage.expected):
            print(f"[SKIP] {stage.title}")
            return
        print(f"\n{'=' * 88}\n[{stage.name}] {stage.title}\n{'=' * 88}")
        print(display_command(stage.command))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{stage.name}.log"
        record = {
            "title": stage.title,
            "status": "running",
            "started_at": iso_now(),
            "command": list(stage.command),
            "command_text": display_command(stage.command),
            "log": str(log_path),
            "expected": [str(path) for path in stage.expected],
        }
        self.state["stages"][stage.name] = record
        self.save_state()
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8", newline="") as log:
                process = subprocess.Popen(
                    stage.command,
                    cwd=REPO_ROOT,
                    env=self.environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                return_code = process.wait()
        except Exception as error:
            record.update(
                status="failed",
                finished_at=iso_now(),
                elapsed_seconds=time.perf_counter() - started,
                error=repr(error),
            )
            self.save_state()
            raise

        record["return_code"] = return_code
        record["elapsed_seconds"] = time.perf_counter() - started
        record["finished_at"] = iso_now()
        missing = [str(path) for path in stage.expected if not file_is_complete(path)]
        if missing:
            record.update(status="failed", missing_outputs=missing)
            self.save_state()
            raise RuntimeError(f"{stage.name} 输出缺失或为空: {missing}")

        if return_code != 0 and stage.allow_audit_threshold_failure:
            audit_path = stage.expected[0]
            try:
                audit = read_json(audit_path)
            except (OSError, ValueError) as error:
                record.update(status="failed", error=f"审计 JSON 无效: {error}")
                self.save_state()
                raise RuntimeError(f"{stage.name} 失败且审计 JSON 无效") from error
            if audit.get("overall_pass") is False:
                record.update(
                    status="threshold_not_met",
                    warning="审计完整执行，但预声明的严格复现阈值未全部通过。",
                )
                self.save_state()
                print("[WARN] 审计文件完整，但严格复现阈值未全部通过；继续采集其余证据。")
                return

        if return_code != 0:
            record.update(status="failed")
            self.save_state()
            raise RuntimeError(
                f"阶段 {stage.name} 返回 {return_code}；详情见 {log_path}"
            )
        record.update(status="completed")
        self.save_state()


def determine_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(DEFAULT_OUTPUT_ROOT) / f"ivc_evidence_{stamp}").resolve()


def require_inputs(args: argparse.Namespace) -> None:
    required = {
        "dataset YAML": Path(args.data),
        "Baseline checkpoint": Path(args.baseline_weights),
        "CA checkpoint": Path(args.ca_weights),
        "Refine checkpoint": Path(args.checkpoint),
        "Refine training directory": Path(args.training_dir),
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("必要输入不存在:\n" + "\n".join(missing))


def require_clean_repository() -> None:
    """Formal evidence must be attributable to one immutable source commit."""
    git_code, _git_commit = capture(["git", "rev-parse", "HEAD"])
    status_code, git_status = capture(["git", "status", "--short"])
    if git_code != 0 or status_code != 0:
        raise RuntimeError("无法读取 Git 状态，拒绝创建不可追溯的正式证据包")
    if git_status:
        raise RuntimeError(
            "正式证据采集要求干净的 Git 工作区；请先提交或清理以下改动:\n" + git_status
        )


def capture(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode, result.stdout.strip()


def write_provenance(args: argparse.Namespace, output_dir: Path) -> None:
    checkpoint_paths = {
        "baseline": Path(args.baseline_weights),
        "ca": Path(args.ca_weights),
        "refine": Path(args.checkpoint),
    }
    git_code, git_commit = capture(["git", "rev-parse", "HEAD"])
    _, git_status = capture(["git", "status", "--short"])
    _, gpu = capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    pip_code, packages = capture([sys.executable, "-m", "pip", "freeze"])
    if pip_code != 0:
        packages = f"pip freeze failed:\n{packages}"
    (output_dir / "python_packages.txt").write_text(packages + "\n", encoding="utf-8")
    hashes = {label: sha256_file(path) for label, path in checkpoint_paths.items()}
    hash_lines = [f"{digest}  {checkpoint_paths[label]}" for label, digest in hashes.items()]
    (output_dir / "checkpoint_sha256.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    provenance = {
        "tool": "collect_ivc_evidence",
        "created_at": iso_now(),
        "command_line": sys.argv,
        "repository": str(REPO_ROOT),
        "git_commit": git_commit if git_code == 0 else None,
        "git_status_short": git_status,
        "python": sys.version,
        "gpu": gpu,
        "omp_num_threads": "1",
        "pythonhashseed": "0",
        "data": str(Path(args.data)),
        "baseline_weights": str(checkpoint_paths["baseline"]),
        "ca_weights": str(checkpoint_paths["ca"]),
        "refine_checkpoint": str(checkpoint_paths["refine"]),
        "checkpoint_sha256": hashes,
        "split": "val",
        "test_used": False,
        "imgsz": 640,
    }
    write_json(output_dir / "provenance.json", provenance)
    environment_lines = [
        f"created_at={provenance['created_at']}",
        f"git_commit={provenance['git_commit']}",
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"gpu={gpu}",
        "OMP_NUM_THREADS=1",
        "PYTHONHASHSEED=0",
        f"data={args.data}",
        f"baseline={args.baseline_weights}",
        f"ca={args.ca_weights}",
        f"refine={args.checkpoint}",
    ]
    (output_dir / "environment.txt").write_text(
        "\n".join(environment_lines) + "\n", encoding="utf-8"
    )


def copy_training_evidence(training_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    required_names = {
        "holdout_metrics.csv",
        "selection.json",
        "acceptance.json",
        "run_manifest.json",
        "train_history.csv",
        "RESULTS.md",
    }
    missing = sorted(name for name in required_names if not (training_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"训练证据缺少必要文件: {missing}")
    allowed_suffixes = {".csv", ".json", ".md", ".txt"}
    for source in sorted(training_dir.iterdir()):
        if source.is_file() and source.suffix.lower() in allowed_suffixes:
            shutil.copy2(source, destination / source.name)


def py_module(module: str, *arguments: object) -> tuple[str, ...]:
    return (sys.executable, "-m", module, *(str(argument) for argument in arguments))


def build_command_stages(args: argparse.Namespace, output_dir: Path) -> list[CommandStage]:
    stages: list[CommandStage] = []
    if args.tests:
        stages.append(
            CommandStage(
                "tests",
                "运行 V3.1.1 协议测试",
                py_module("pytest", "-q", "tests/V3_1_1"),
            )
        )

    main_csv = output_dir / "main_validation.csv"
    stages.append(
        CommandStage(
            "main_validation",
            "采集 Baseline 与 CA 主结果",
            py_module(
                "myscripts.paper_visuals.collect_validation_metrics",
                "--model",
                f"Baseline={args.baseline_weights}",
                "--model",
                f"CA={args.ca_weights}",
                "--data",
                args.data,
                "--imgsz",
                640,
                "--batch",
                8,
                "--device",
                args.device,
                "--workers",
                args.workers,
                "--project",
                output_dir / "main_validation_runs",
                "--output-csv",
                main_csv,
            ),
            (main_csv,),
        )
    )

    validations = (
        ("reproduction_fp32_batch8", "正式 FP32 batch=8 Refine 验证", 8, "--no-amp"),
        ("reproduction_amp_batch8", "AMP batch=8 稳健性复核", 8, "--amp"),
        ("reproduction_fp32_batch1", "FP32 batch=1 稳健性复核", 1, "--no-amp"),
    )
    for name, title, batch, amp_flag in validations:
        destination = output_dir / name
        stages.append(
            CommandStage(
                name,
                title,
                py_module(
                    "myscripts.V3_1_1.validate_refine_v311",
                    "--checkpoint",
                    args.checkpoint,
                    "--ca-weights",
                    args.ca_weights,
                    "--data",
                    args.data,
                    "--imgsz",
                    640,
                    "--batch",
                    batch,
                    "--device",
                    args.device,
                    "--workers",
                    args.workers,
                    amp_flag,
                    "--output-dir",
                    destination,
                ),
                (
                    destination / "val_metrics.csv",
                    destination / "val_diagnostics.json",
                    destination / "validation_audit.json",
                ),
            )
        )

    audit_dir = output_dir / "reproduction_audit"
    stages.append(
        CommandStage(
            "reproduction_audit",
            "汇总三条 Refine 复核路径",
            py_module(
                "myscripts.V3_1_1.audit_reproductions_v311",
                "--training-dir",
                args.training_dir,
                "--amp-batch8-dir",
                output_dir / "reproduction_amp_batch8",
                "--fp32-batch8-dir",
                output_dir / "reproduction_fp32_batch8",
                "--fp32-batch1-dir",
                output_dir / "reproduction_fp32_batch1",
                "--output-dir",
                audit_dir,
            ),
            (audit_dir / "reproduction_audit.json", audit_dir / "reproduction_audit.md"),
            allow_audit_threshold_failure=True,
        )
    )

    complexity_csv = output_dir / "complexity_baseline_ca.csv"
    stages.append(
        CommandStage(
            "complexity_baseline_ca",
            "采集 Baseline 与 CA 参数量、GFLOPs 和 validator 延迟",
            py_module(
                "myscripts.paper_visuals.collect_model_complexity",
                "--model",
                f"Baseline={args.baseline_weights}",
                "--model",
                f"CA={args.ca_weights}",
                "--data",
                args.data,
                "--imgsz",
                640,
                "--batch",
                1,
                "--device",
                args.device,
                "--workers",
                args.workers,
                "--project",
                output_dir / "complexity_runs",
                "--output-csv",
                complexity_csv,
            ),
            (complexity_csv,),
        )
    )

    if args.smoke:
        smoke_profile = output_dir / "smoke_profile_refine"
        stages.append(
            CommandStage(
                "smoke_profile_refine",
                "用 20 张图检查完整链性能采集器",
                py_module(
                    "myscripts.V3_1_1.profile_refine_v311",
                    "--checkpoint",
                    args.checkpoint,
                    "--ca-weights",
                    args.ca_weights,
                    "--data",
                    args.data,
                    "--imgsz",
                    640,
                    "--batch",
                    1,
                    "--device",
                    args.device,
                    "--workers",
                    args.workers,
                    "--no-amp",
                    "--warmup",
                    20,
                    "--max-images",
                    20,
                    "--output-dir",
                    smoke_profile,
                ),
                (smoke_profile / "profile_per_image.csv", smoke_profile / "profile_summary.json"),
            )
        )

    profile_dir = output_dir / "profile_refine_fp32_batch1"
    stages.append(
        CommandStage(
            "profile_refine_fp32_batch1",
            "采集完整验证集的 detector+NMS+Refine 同步性能",
            py_module(
                "myscripts.V3_1_1.profile_refine_v311",
                "--checkpoint",
                args.checkpoint,
                "--ca-weights",
                args.ca_weights,
                "--data",
                args.data,
                "--imgsz",
                640,
                "--batch",
                1,
                "--device",
                args.device,
                "--workers",
                args.workers,
                "--no-amp",
                "--warmup",
                OFFICIAL_WARMUP_PASSES,
                "--output-dir",
                profile_dir,
            ),
            (profile_dir / "profile_per_image.csv", profile_dir / "profile_summary.json"),
        )
    )

    comparative_dir = output_dir / "profile_comparative_fp32_batch1"
    stages.append(
        CommandStage(
            "profile_comparative_fp32_batch1",
            "以三轮平衡顺序和独立进程比较 Baseline、CA 与 CA+Refine 性能",
            py_module(
                "myscripts.V3_1_1.profile_comparative_v311",
                "--baseline-weights",
                args.baseline_weights,
                "--ca-weights",
                args.ca_weights,
                "--refine-profile-summary",
                profile_dir / "profile_summary.json",
                "--data",
                args.data,
                "--imgsz",
                640,
                "--batch",
                1,
                "--device",
                args.device,
                "--workers",
                args.workers,
                "--no-amp",
                "--warmup",
                OFFICIAL_WARMUP_PASSES,
                "--repeats",
                3,
                "--output-dir",
                comparative_dir,
            ),
            (
                comparative_dir / "comparative_latency.csv",
                comparative_dir / "comparative_profile.json",
                comparative_dir / "comparative_per_image.csv",
                comparative_dir / "comparative_repeat_summary.csv",
            ),
        )
    )

    if args.smoke:
        smoke_qualitative = output_dir / "smoke_qualitative"
        stages.append(
            CommandStage(
                "smoke_qualitative",
                "用 16 张图检查四路定性导出器",
                py_module(
                    "myscripts.V3_1_1.export_qualitative_v311",
                    "--baseline-weights",
                    args.baseline_weights,
                    "--ca-weights",
                    args.ca_weights,
                    "--checkpoint",
                    args.checkpoint,
                    "--data",
                    args.data,
                    "--imgsz",
                    640,
                    "--batch",
                    8,
                    "--device",
                    args.device,
                    "--workers",
                    args.workers,
                    "--no-amp",
                    "--max-images",
                    16,
                    "--copy-images",
                    "--exist-ok",
                    "--output-dir",
                    smoke_qualitative,
                ),
                (
                    smoke_qualitative / "manifest_all.csv",
                    smoke_qualitative / "export_audit.json",
                ),
            )
        )

    qualitative_dir = output_dir / "qualitative_predictions"
    stages.append(
        CommandStage(
            "qualitative_predictions",
            "导出完整验证集的 GT/Baseline/CA/CA+Refine 定性数据",
            py_module(
                "myscripts.V3_1_1.export_qualitative_v311",
                "--baseline-weights",
                args.baseline_weights,
                "--ca-weights",
                args.ca_weights,
                "--checkpoint",
                args.checkpoint,
                "--data",
                args.data,
                "--imgsz",
                640,
                "--batch",
                8,
                "--device",
                args.device,
                "--workers",
                args.workers,
                "--no-amp",
                "--copy-images",
                "--exist-ok",
                "--output-dir",
                qualitative_dir,
            ),
            (qualitative_dir / "manifest_all.csv", qualitative_dir / "export_audit.json"),
        )
    )

    if args.h1h2:
        for method, weights in (
            ("baseline", args.baseline_weights),
            ("ca", args.ca_weights),
        ):
            destination = output_dir / "h1h2" / method
            stages.append(
                CommandStage(
                    f"h1h2_{method}",
                    f"采集 {method} 的 DFL 溢出与 FPN 正样本分配统计",
                    py_module(
                        "myscripts.check_h1h2_stats",
                        "--method",
                        method,
                        "--model",
                        weights,
                        "--data",
                        args.data,
                        "--split",
                        "val",
                        "--passes",
                        args.h1h2_passes,
                        "--batch",
                        8,
                        "--imgsz",
                        640,
                        "--device",
                        args.device,
                        "--workers",
                        args.workers,
                        "--project",
                        output_dir / "h1h2",
                        "--name",
                        method,
                        "--exist-ok",
                    ),
                    (
                        destination / "h1h2_data.json",
                        destination / "h1h2_report.md",
                        destination / "h1h2_run_config.json",
                    ),
                )
            )
    return stages


def audit_protocol_outputs(args: argparse.Namespace, output_dir: Path) -> list[str]:
    """Check the locked protocol without re-running a model."""
    warnings: list[str] = []
    with (output_dir / "main_validation.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        main_rows = list(csv.DictReader(handle))
    complete_ap_columns = {f"ap{threshold}" for threshold in range(50, 100, 5)}
    if not main_rows or not complete_ap_columns.issubset(main_rows[0]):
        raise RuntimeError("主结果CSV缺少AP50至AP95的完整5点阈值序列")
    main_models = {row.get("method"): row.get("weights") for row in main_rows}
    if main_models != {
        "Baseline": str(Path(args.baseline_weights)),
        "CA": str(Path(args.ca_weights)),
    }:
        raise RuntimeError(f"主结果模型身份不正确或行数异常: {main_models}")

    formal = read_json(output_dir / "reproduction_fp32_batch8" / "validation_audit.json")
    with (output_dir / "reproduction_fp32_batch8" / "val_metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        formal_rows = list(csv.DictReader(handle))
    if not formal_rows or not complete_ap_columns.issubset(formal_rows[0]):
        raise RuntimeError("正式Refine验证CSV缺少AP50至AP95的完整5点阈值序列")
    if not (
        formal.get("split") == "val"
        and formal.get("test_used") is False
        and formal.get("imgsz") == 640
        and formal.get("batch") == 8
        and formal.get("amp") is False
        and formal.get("identity_pass") is True
        and formal.get("rerun_nms") is False
        and formal.get("reproduction_pass") is True
        and formal.get("ca_weights") == str(Path(args.ca_weights))
        and formal.get("checkpoint") == str(Path(args.checkpoint))
    ):
        raise RuntimeError("正式 Refine 验证不符合 val/FP32/batch8/imgsz640/恒等协议")
    profile = read_json(output_dir / "profile_refine_fp32_batch1" / "profile_summary.json")
    if not (
        profile.get("split") == "val"
        and profile.get("test_used") is False
        and profile.get("protocol_version") == EVIDENCE_PROTOCOL_VERSION
        and profile.get("imgsz") == 640
        and profile.get("batch") == 1
        and profile.get("amp") is False
        and profile.get("warmup_passes") == OFFICIAL_WARMUP_PASSES
        and profile.get("weights_unchanged") is True
        and profile.get("refiner_flop_profile_proposals") is not None
        and "peak_allocated_gib" in profile.get("gpu_memory", {})
    ):
        raise RuntimeError("完整链性能文件不符合 val/FP32/batch1/imgsz640/权重不变协议")
    comparative = read_json(
        output_dir / "profile_comparative_fp32_batch1" / "comparative_profile.json"
    )
    if not (
        comparative.get("split") == "val"
        and comparative.get("test_used") is False
        and comparative.get("imgsz") == 640
        and comparative.get("batch") == 1
        and comparative.get("amp") is False
        and comparative.get("protocol_version") == EVIDENCE_PROTOCOL_VERSION
        and comparative.get("repeat_count") == 3
        and comparative.get("warmup_passes") == OFFICIAL_WARMUP_PASSES
        and comparative.get("same_images_and_shapes") is True
        and comparative.get("same_image_order") is True
        and comparative.get("ca_proposals_match_refine_profile") is True
        and comparative.get("external_refine_paths_match") is True
        and comparative.get("external_refine_proposals_match") is True
        and comparative.get("worker_process_isolation") is True
        and comparative.get("baseline", {}).get("weights_unchanged") is True
        and comparative.get("ca", {}).get("weights_unchanged") is True
        and comparative.get("refine", {}).get("weights_unchanged") is True
        and comparative.get("refine", {}).get("flop_profile_proposals") is not None
        and "proposal_count" in comparative.get("refine", {})
        and "isolated_peak_gpu_memory_gib" in comparative.get("refine", {})
    ):
        raise RuntimeError("统一 Baseline/CA/CA+Refine 性能文件不符合锁定协议")
    if comparative.get("latency_stability_pass") is not True:
        warnings.append("三轮平衡顺序延迟的轮间波动超过预声明5%阈值，效率结果暂不进入论文。")
    if comparative.get("memory_stability_pass") is not True:
        warnings.append("独立进程峰值显存的轮间波动超过预声明2%阈值，显存结果暂不进入论文。")
    if comparative.get("ca_refine_coarse_consistency_pass") is not True:
        warnings.append("独立CA与Refine链内部coarse耗时差超过预声明5%阈值，Refine开销暂不进入论文。")
    qualitative = read_json(output_dir / "qualitative_predictions" / "export_audit.json")
    if not (
        qualitative.get("split") == "val"
        and qualitative.get("test_used") is False
        and qualitative.get("imgsz") == 640
        and qualitative.get("amp") is False
        and qualitative.get("identity_pass") is True
        and qualitative.get("weights_unchanged") is True
        and qualitative.get("rerun_nms") is False
    ):
        raise RuntimeError("定性导出不符合锁定协议或恒等/权重审计未通过")
    reproduction = read_json(output_dir / "reproduction_audit" / "reproduction_audit.json")
    if reproduction.get("test_used") is not False or reproduction.get("hash_pass") is not True:
        raise RuntimeError("三协议复现审计没有确认 test 未使用或权重哈希一致")
    if reproduction.get("overall_pass") is False:
        warnings.append(
            "严格复现审计 overall_pass=false；保留原 5e-4 阈值和实际差值，不影响证据采集完整性。"
        )
    if args.h1h2:
        for method, weights in (
            ("baseline", args.baseline_weights),
            ("ca", args.ca_weights),
        ):
            config = read_json(output_dir / "h1h2" / method / "h1h2_run_config.json")
            if not (
                config.get("method") == method
                and config.get("imgsz") == 640
                and config.get("model") == str(Path(weights).resolve())
                and config.get("strict_stats") is True
                and config.get("split") == "val"
                and config.get("passes") == args.h1h2_passes
                and config.get("mode") == "frozen_eval_assigner_statistics"
                and config.get("optimization_steps") == 0
                and config.get("checkpoint_saving") is False
                and config.get("model_state_unchanged") is True
                and config.get("checkpoint_file_unchanged") is True
            ):
                raise RuntimeError(f"H1/H2 {method} 运行配置不符合锁定协议")
    return warnings


def write_file_inventory(output_dir: Path) -> None:
    excluded = {"file_manifest.txt", "evidence_sha256.txt"}
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    manifest_lines = [path.relative_to(output_dir).as_posix() for path in files]
    (output_dir / "file_manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    hash_lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in files
    ]
    (output_dir / "evidence_sha256.txt").write_text(
        "\n".join(hash_lines) + "\n", encoding="utf-8"
    )


def create_archive(output_dir: Path) -> Path:
    archive = Path(str(output_dir) + ".tar.gz")
    temporary = Path(str(archive) + ".tmp")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=6) as handle:
            handle.add(output_dir, arcname=output_dir.name)
        temporary.replace(archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return archive


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"
    require_clean_repository()
    output_dir = determine_output_dir(args)
    if output_dir.exists() and not args.resume:
        parser.error(f"输出目录已存在，避免混入旧结果: {output_dir}；继续时请增加 --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"IVC 证据输出目录: {output_dir}")
    collector = EvidenceCollector(output_dir, resume=args.resume)
    collector.lock_run_identity(build_run_identity(args))
    collector.state["arguments"] = vars(args) | {"output_dir": str(output_dir)}
    collector.save_state()
    require_inputs(args)

    collector.run_action(
        "provenance",
        "记录环境、代码版本和三个 checkpoint 哈希",
        lambda: write_provenance(args, output_dir),
        (
            output_dir / "provenance.json",
            output_dir / "environment.txt",
            output_dir / "checkpoint_sha256.txt",
            output_dir / "python_packages.txt",
        ),
    )
    collector.run_action(
        "training_holdout",
        "复制既有 Refine holdout 与检查点选择证据（不重新训练）",
        lambda: copy_training_evidence(
            Path(args.training_dir), output_dir / "training_holdout"
        ),
        (
            output_dir / "training_holdout" / "holdout_metrics.csv",
            output_dir / "training_holdout" / "selection.json",
            output_dir / "training_holdout" / "acceptance.json",
            output_dir / "training_holdout" / "run_manifest.json",
        ),
    )

    stages = build_command_stages(args, output_dir)
    collector.state["planned_stages"] = [stage.name for stage in stages]
    collector.save_state()
    try:
        for stage in stages:
            collector.run_command(stage)
        warnings = audit_protocol_outputs(args, output_dir)
        statuses = {
            name: record.get("status")
            for name, record in collector.state.get("stages", {}).items()
        }
        collector.state.update(
            status="completed",
            completed_at=iso_now(),
            warnings=warnings,
        )
        collector.save_state()
        summary = {
            "tool": "collect_ivc_evidence",
            "status": "completed",
            "completed_at": collector.state["completed_at"],
            "output_dir": str(output_dir),
            "archive_requested": bool(args.archive),
            "protocol": collector.state["protocol"],
            "stage_status": statuses,
            "warnings": warnings,
            "notes": [
                "三条复核路径复用同一个 Refine checkpoint，不是三个随机种子。",
                "H1/H2 是固定checkpoint、单一val split、零优化步骤的只读统计。",
                "Baseline、CA和CA+Refine以三轮拉丁方顺序、独立进程和同一同步链路输出延迟与显存。",
                "主验证CSV和Refine验证CSV均包含AP50至AP95的完整5点阈值序列。",
                "正式论文主结果取 reproduction_fp32_batch8。",
            ],
        }
        write_json(output_dir / "collection_summary.json", summary)
        write_file_inventory(output_dir)
        archive = create_archive(output_dir) if args.archive else None
    except Exception:
        collector.state.update(status="failed", failed_at=iso_now())
        collector.save_state()
        if (output_dir / "evidence_sha256.txt").exists():
            write_file_inventory(output_dir)
        print(f"\n采集中断。修复问题后使用同一目录继续：\n  --output-dir {output_dir} --resume")
        raise

    print("\n" + "=" * 88)
    print("IVC 数据采集完成")
    print(f"数据目录: {output_dir}")
    if archive is not None:
        print(f"压缩包: {archive}")
    if warnings:
        print("保留的审计提示:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
