#!/usr/bin/env bash
# Run one experiment stage with the existing Python environment on the workspace server.
set -euo pipefail
cd /workspace/code

usage() {
    echo "Usage: bash myscripts/run_wire_seg_server.sh ca|refine|smoke"
    echo "       bash myscripts/run_wire_seg_server.sh cache|audit [train|val|test]"
    echo "       bash myscripts/run_wire_seg_server.sh train|bench [target|points|boxes|axes]"
    echo "       Training controls: WIRE_BATCH, WIRE_BATCH_PIXELS, WIRE_WORKERS, WIRE_AMP (0 or 1)"
    echo "       bash myscripts/run_wire_seg_server.sh eval|predict [target|points|boxes|axes] [val|test]"
}
stage=${1:-help}
case "$stage" in
    help|-h|--help) usage; exit 0 ;;
    smoke) exec python -m unittest discover -s tests -p test_wire_instance_seg.py -v ;;
    ca|refine|cache|audit|train|bench|eval|predict) ;;
    *) usage >&2; exit 2 ;;
esac

work=/workspace/workdir
run="$work/wire_seg"
data=$(python -m myscripts.workspace_ttpla root)
ca=${CA_WEIGHTS:-"$work/yolo11_obb_640_811_ca/weights/best.pt"}
refine_dir=${REFINE_DIR:-"$work/refine_v311_ca_seed0_$(date +%Y%m%d-%H%M%S)"}
gpu=${GPU_ID:-0}

case "$stage" in
    ca)
        exec python train_yolo11_obb.py
        ;;
    refine)
        config=$(python -m myscripts.workspace_ttpla yaml)
        exec python -m myscripts.V3_1_1.train_refine_v311 --data "$config" \
            --ca-weights "$ca" --output-dir "$refine_dir" --device "$gpu"
        ;;
    cache|audit)
        split=${2:-val}
        ;;
    train|bench|eval|predict)
        prompt=${2:-axes}
        case "$prompt" in target|points|boxes|axes) ;; *) usage >&2; exit 2 ;; esac
        split=${3:-val}
        if [[ "$stage" == train || "$stage" == bench ]]; then split=train; fi
        ;;
esac
case "$split" in train|val|test) ;; *) usage >&2; exit 2 ;; esac
images="$data/images/$split"
jsons="$data/jsons/$split"
cache="$run/proposals/$split.json"

case "$stage" in
    cache)
        refine=${REFINE_WEIGHTS:-"$(python -m myscripts.workspace_ttpla refine-dir)/checkpoints/best.pt"}
        exec python train_wire_seg.py cache --images "$images" --ca "$ca" --refine "$refine" \
            --device "$gpu" --output "$cache"
        ;;
    audit)
        exec python train_wire_seg.py audit --images "$images" --jsons "$jsons" --cache "$cache" \
            --margin 12 --output "$run/audit/$split.json"
        ;;
    train|bench)
        batch=${WIRE_BATCH:-64}
        workers=${WIRE_WORKERS:-4}
        pixels=${WIRE_BATCH_PIXELS:-2097152}
        precision=(--amp)
        if [[ ${WIRE_AMP:-1} == 0 ]]; then precision=(--no-amp); fi
        schedule=(--epochs 15)
        output="$run/$prompt/mask.pt"
        if [[ "$stage" == bench ]]; then
            schedule=(--epochs 1 --limit 512)
            output="$run/bench/${prompt}_b${batch}_w${workers}_px${pixels}_amp${WIRE_AMP:-1}.pt"
        fi
        exec python train_wire_seg.py train --images "$images" --jsons "$jsons" --cache "$cache" \
            --prompt "$prompt" --boxes refined --margin 12 "${schedule[@]}" --seed 0 --device "cuda:$gpu" \
            --batch-size "$batch" --workers "$workers" --max-batch-pixels "$pixels" "${precision[@]}" \
            --output "$output"
        ;;
    eval)
        exec python train_wire_seg.py eval --images "$images" --jsons "$jsons" --cache "$cache" \
            --weights "$run/$prompt/mask.pt" --device "cuda:$gpu" --swap-target \
            --output "$run/$prompt/metrics_$split.json"
        ;;
    predict)
        exec python train_wire_seg.py predict --images "$images" --cache "$cache" \
            --weights "$run/$prompt/mask.pt" --device "cuda:$gpu" --output "$run/$prompt/masks_$split"
        ;;
esac
