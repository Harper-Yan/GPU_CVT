#!/usr/bin/env bash
# Run gpu_cvt for modes 0, 1, 2 on a mesh (100 iters each).
# Usage: ./scripts/run_three_modes.sh <path_to_mesh.obj>
# Example: ./scripts/run_three_modes.sh objs/stanford-bunny.obj

set -e
MESH="${1:?Usage: $0 <mesh.obj>}"
BIN="${BIN:-./build/gpu_cvt}"
if [[ ! -f "$MESH" ]]; then
  echo "Mesh not found: $MESH"
  exit 1
fi
if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN"
  exit 1
fi

echo "=== Mode 0 (baseline) ==="
"$BIN" "$MESH" 0
echo ""
echo "=== Mode 1 (freeze, 5-tier) ==="
"$BIN" "$MESH" 1
echo ""
echo "=== Mode 2 (freeze_tiered, 6-tier) ==="
"$BIN" "$MESH" 2
echo ""
echo "Done. Eval CSVs: experiments/output/gpucvt/<mesh>/ experiments/output/freeze/<mesh>/ experiments/output/freeze_tiered/<mesh>/"
