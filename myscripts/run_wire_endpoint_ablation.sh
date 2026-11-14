#!/usr/bin/env bash
# One new training arm; refresh both evaluations using the existing split/cache.
set -euo pipefail
cd /workspace/code
run=/workspace/workdir/wire_seg
out="$run/endpoint_ablation_val_$(date +%Y%m%d_%H%M%S)"
test -f "$run/axes/mask.pt"
test -f "$run/proposals/train.json"
test -f "$run/proposals/val.json"
mkdir -p "$out"
export WIRE_BATCH=${WIRE_BATCH:-64}
export WIRE_WORKERS=${WIRE_WORKERS:-4}
export WIRE_BATCH_PIXELS=${WIRE_BATCH_PIXELS:-2097152}
export WIRE_AMP=${WIRE_AMP:-1}
export PYTHONUNBUFFERED=1
(
    bash myscripts/run_wire_seg_server.sh smoke
    bash myscripts/run_wire_seg_server.sh train axes_infinite
    for prompt in axes axes_infinite; do
        bash myscripts/run_wire_seg_server.sh eval "$prompt" val
        bash myscripts/run_wire_seg_server.sh threshold-scan "$prompt" val
    done
) 2>&1 | tee "$out/run.log"
python -m myscripts.wire_seg.endpoint_ablation --run-dir "$run" --output-dir "$out"
