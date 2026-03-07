#!/usr/bin/env bash
# Run modes 0, 1, 2 for every .obj in objs/, then plot for each mesh.
# Usage: ./scripts/run_all_meshes_three_modes.sh [objs_dir]
# Default objs_dir: objs

set -e
OBJS_DIR="${1:-objs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_three_modes.sh"
PLOT_SCRIPT="$SCRIPT_DIR/plot_three_modes.py"

if [[ ! -d "$ROOT/$OBJS_DIR" ]]; then
  echo "Directory not found: $ROOT/$OBJS_DIR"
  exit 1
fi

cd "$ROOT"
for obj in "$OBJS_DIR"/*.obj; do
  [[ -f "$obj" ]] || continue
  mesh="$(basename "$obj" .obj)"
  echo "=============================================="
  echo "Mesh: $mesh ($obj)"
  echo "=============================================="
  "$RUN_SCRIPT" "$obj"
  python3 "$PLOT_SCRIPT" --mesh "$mesh" --output-dir experiments/output --no-show
  echo ""
done
echo "Done. Plots: experiments/plots_eval/<mesh>/"
