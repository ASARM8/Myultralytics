"""Generate data-dependent figures 3--6 and tables whose required inputs are present in one YAML configuration.

Figures 1 and 2 are editable PowerPoint explanatory figures. Build them separately with
``architecture/build_ca_refine_architecture.mjs`` and ``architecture/build_fig2_geometric_reachability.mjs``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str | Path, base: Path) -> Path:
    """Resolve a configuration path relative to the configuration file."""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def run(module: str, arguments: list[str]) -> None:
    """Run one generator using the active local Python interpreter."""
    command = [sys.executable, "-m", module, *arguments]
    print("RUN", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Fail when a configured input path is missing")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    output_dir = resolve(config.get("output_dir", "outputs"), base)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = config.get("figures", {})

    fig3 = figures.get("fig3", {})
    if fig3.get("baseline_json") and fig3.get("ca_json"):
        inputs = [resolve(fig3[key], base) for key in ("baseline_json", "ca_json")]
        if all(path.exists() for path in inputs):
            run(
                "myscripts.paper_visuals.generate_fig3_dfl_overflow",
                ["--baseline-json", str(inputs[0]), "--ca-json", str(inputs[1]), "--output-dir", str(output_dir)],
            )
        elif args.strict:
            raise FileNotFoundError(inputs)

    fig4 = figures.get("fig4", {})
    fig4_args: list[str] = []
    for label, value in fig4.get("series", {}).items():
        path = resolve(value, base)
        if path.exists():
            fig4_args.extend(["--series", f"{label}={path}"])
        elif args.strict:
            raise FileNotFoundError(path)
    for value in fig4.get("tidy_csv", []):
        path = resolve(value, base)
        if path.exists():
            fig4_args.extend(["--tidy-csv", str(path)])
        elif args.strict:
            raise FileNotFoundError(path)
    for label, offset in fig4.get("epoch_offset", {}).items():
        fig4_args.extend(["--epoch-offset", f"{label}={offset}"])
    if fig4_args:
        fig4_args.extend(["--smooth-window", str(fig4.get("smooth_window", 5)), "--output-dir", str(output_dir)])
        run("myscripts.paper_visuals.generate_fig4_training_curves", fig4_args)

    fig5 = figures.get("fig5", {})
    if fig5.get("baseline_json") and fig5.get("ca_json"):
        inputs = [resolve(fig5[key], base) for key in ("baseline_json", "ca_json")]
        if all(path.exists() for path in inputs):
            run(
                "myscripts.paper_visuals.generate_fig5_level_distribution",
                ["--baseline-json", str(inputs[0]), "--ca-json", str(inputs[1]), "--output-dir", str(output_dir)],
            )
        elif args.strict:
            raise FileNotFoundError(inputs)

    fig6 = figures.get("fig6", {})
    if fig6.get("manifest"):
        manifest = resolve(fig6["manifest"], base)
        if manifest.exists():
            fig6_args = ["--manifest", str(manifest), "--output-dir", str(output_dir)]
            if fig6.get("show_confidence", False):
                fig6_args.append("--show-confidence")
            run("myscripts.paper_visuals.generate_fig6_qualitative", fig6_args)
        elif args.strict:
            raise FileNotFoundError(manifest)

    tables = config.get("tables", {})
    table_args = ["--output-dir", str(output_dir / "tables")]
    option_map = {
        "baseline_h1h2": "--baseline-h1h2",
        "ca_h1h2": "--ca-h1h2",
        "dataset_summary": "--dataset-summary",
        "table07_extra": "--table07-extra",
        "manual_dir": "--manual-dir",
    }
    for key, option in option_map.items():
        if not tables.get(key):
            continue
        path = resolve(tables[key], base)
        if path.exists():
            table_args.extend([option, str(path)])
        elif args.strict:
            raise FileNotFoundError(path)
    if tables.get("dataset_name"):
        table_args.extend(["--dataset-name", str(tables["dataset_name"])])
    run("myscripts.paper_visuals.build_tables", table_args)


if __name__ == "__main__":
    main()
