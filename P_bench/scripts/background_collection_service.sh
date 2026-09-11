#!/usr/bin/env bash
set -euo pipefail

UNIT=pbench-background-collection.service

usage() {
    echo "usage: $0 start <manifest> | stop | status <manifest>" >&2
    exit 2
}

load_manifest() {
    local loader=${EGOCONSEQ_HABITAT_PYTHON:-python}
    mapfile -t MANIFEST_VALUES < <(
        "$loader" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
value = json.loads(path.read_text())
paths = value["paths"]
for item in (
    path, value["output_root"], paths["repository"], paths["python"],
    paths["r2r_train_episodes"], paths["mp3d_root"],
    paths["gs_data_root"], paths["gs_source_manifest"],
):
    print(item)
' "$1"
    )
}

case ${1:-} in
    start)
        [[ $# -eq 2 ]] || usage
        load_manifest "$2"
        manifest=${MANIFEST_VALUES[0]}
        output_root=${MANIFEST_VALUES[1]}
        repository=${MANIFEST_VALUES[2]}
        python=${MANIFEST_VALUES[3]}
        mkdir -p "$output_root/controller"
        log="$output_root/controller/systemd.log"
        systemd-run --user --no-block --collect --unit="$UNIT" \
            --property=Type=exec \
            --property=Restart=on-failure \
            --property=RestartSec=5 \
            --property=KillMode=mixed \
            --property=SuccessExitStatus=130 \
            --property=TimeoutStopSec=210 \
            --property="StandardOutput=append:$log" \
            --property="StandardError=append:$log" \
            --working-directory="$repository" \
            --setenv="EGOCONSEQ_HABITAT_PYTHON=$python" \
            --setenv="EGOCONSEQ_R2R_TRAIN_EPISODES=${MANIFEST_VALUES[4]}" \
            --setenv="EGOCONSEQ_MP3D_ROOT=${MANIFEST_VALUES[5]}" \
            --setenv="EGOCONSEQ_GS_ROOT=${MANIFEST_VALUES[6]}" \
            --setenv="EGOCONSEQ_GS_TRAIN_MANIFEST=${MANIFEST_VALUES[7]}" \
            --setenv=PYTHONUNBUFFERED=1 \
            "$python" "$repository/scripts/run_background_collection.py" \
            run --manifest "$manifest"
        ;;
    stop)
        [[ $# -eq 1 ]] || usage
        systemctl --user stop "$UNIT"
        ;;
    status)
        [[ $# -eq 2 ]] || usage
        load_manifest "$2"
        exec "${MANIFEST_VALUES[3]}" \
            "${MANIFEST_VALUES[2]}/scripts/run_background_collection.py" \
            status --manifest "${MANIFEST_VALUES[0]}"
        ;;
    *)
        usage
        ;;
esac
