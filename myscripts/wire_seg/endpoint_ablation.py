"""Collect the finite/infinite neighbor-axis experiment and make a download ZIP."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def collect(run_dir, output_dir):
    import torch  # Existing server environment; no model inference here.

    run, out = Path(run_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = ("axes", "axes_infinite")
    reports, scans, checkpoints, files = {}, {}, {}, []
    for arm in arms:
        for filename, dest in (("metrics_val.json", reports), ("threshold_scan_val.json", scans)):
            path = run / arm / filename
            dest[arm] = json.loads(path.read_text(encoding="utf-8"))
            if dest[arm]["prompt"] != arm:
                raise ValueError(f"Wrong prompt in {path}")
            files.append(path)
        checkpoints[arm] = torch.load(run / arm / "mask.pt", map_location="cpu", weights_only=True)
    keys = ("margin", "boxes", "detector", "epoch", "seed", "batch_size", "max_batch_pixels",
            "precision", "train_sources", "train_videos")
    for key in keys:
        if checkpoints[arms[0]][key] != checkpoints[arms[1]][key]:
            raise ValueError(f"Training metadata mismatch: {key}; align both arms before comparison")
    for key in ("cache_sha256", "mask_thresholds", "tiles", "ground_truth_instances", "coarse_obb_match_iou"):
        if scans[arms[0]][key] != scans[arms[1]][key]:
            raise ValueError(f"Evaluation protocol mismatch: {key}")
    for key in ("tiles", "ground_truth_instances", "swap_target_pairs"):
        if reports[arms[0]][key] != reports[arms[1]][key]:
            raise ValueError(f"Evaluation cohort mismatch: {key}")
    rows = []
    def add(condition, metric, finite, infinite):
        rows.append({"condition": condition, "metric": metric, "finite": finite, "infinite": infinite,
                     "finite_minus_infinite": finite-infinite if finite is not None and infinite is not None else None})
    for name in ("mask_mAP50_95", "swap_both_correct_fraction"):
        add("mask_threshold=0.5", name, *(reports[a][name] for a in arms))
    for cohort in ("matched", "matched_with_neighbors", "matched_with_neighbor_within_15deg"):
        for name in ("mean_instance_iou", "target_pixel_recall", "neighbor_pixel_leakage"):
            add("mask_threshold=0.5", f"{cohort}.{name}", *(reports[a][cohort][name] for a in arms))
    finite_scan = {r["mask_threshold"]: r for r in scans[arms[0]]["results"]}
    infinite_scan = {r["mask_threshold"]: r for r in scans[arms[1]]["results"]}
    for threshold, f in finite_scan.items():
        i = infinite_scan[threshold]
        add(f"mask_threshold={threshold:g}", "mask_mAP50_95", f["mask_mAP50_95"], i["mask_mAP50_95"])
        for key in ("target_pixel_recall", "neighbor_pixel_leakage"):
            add(f"mask_threshold={threshold:g}", "matched_with_neighbors."+key,
                f["matched_with_neighbors"][key], i["matched_with_neighbors"][key])
        for cutoff in ("0.50", "0.75"):
            add(f"mask_threshold={threshold:g}", "successful_pairs_iou"+cutoff,
                f["neighbor_pairs"]["all"]["by_mask_iou"][cutoff][arms[0]]["successful_pairs"],
                i["neighbor_pairs"]["all"]["by_mask_iou"][cutoff][arms[1]]["successful_pairs"])
    with (out/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "intervention": "Only neighbor axial endpoint distance removed; finite target prompt unchanged",
        "finite_prompt": "exp(-(v^2 + max(abs(u)-L/2,0)^2)/(2*sigma^2))",
        "infinite_prompt": "exp(-v^2/(2*sigma^2))",
        "training_metadata": {k: checkpoints[arms[0]][k] for k in keys},
        "protocol": {k: scans[arms[0]][k] for k in ("cache_sha256", "mask_thresholds", "tiles")},
        "rows": rows,
        "units": "Fractions are 0..1; multiply their difference by 100 for percentage points. Pair metrics are counts.",
        "weights_sha256": {a: scans[a]["weights_sha256"] for a in arms},
        "source_sha256": {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (out/"comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = Path(str(out)+".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as pack:
        for p in files:
            pack.write(p, p.relative_to(run).as_posix())
        for p in sorted(out.iterdir()):
            if p.is_file(): pack.write(p, "comparison/"+p.name)
    print(f"[endpoint-ablation] summary: {out / 'comparison.json'}", flush=True)
    print(f"[endpoint-ablation] download: {archive}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", required=True)
    cli.add_argument("--output-dir", required=True)
    args = cli.parse_args()
    collect(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
