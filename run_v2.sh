#!/usr/bin/env bash
set -euo pipefail
python -u -m paraseedbench.run_v2 --config "${1:-configs/main_v2.lock.yaml}"
