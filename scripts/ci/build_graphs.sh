#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-artifacts}
MODE=${2:-both}
TEMPLATE=${3:-scripts/pages_index.html}
BRAGG_IMAGE=${BRAGG_IMAGE:-IMG.jpg}
BRAGG_BIG_CIRCLE_THRESH=${BRAGG_BIG_CIRCLE_THRESH:-10}
BRAGG_SMALL_DOT_THRESH=${BRAGG_SMALL_DOT_THRESH:-50}
BRAGG_SMALL_DOT_THRESH_OUTER=${BRAGG_SMALL_DOT_THRESH_OUTER:-30}
BRAGG_MIN_SPOT_AREA=${BRAGG_MIN_SPOT_AREA:-10}
BRAGG_MIN_CIRCULARITY=${BRAGG_MIN_CIRCULARITY:-0.2}
BRAGG_PHYS_Y_MM=${BRAGG_PHYS_Y_MM:-75}
BRAGG_PHYS_X_MM=${BRAGG_PHYS_X_MM:-55}
BRAGG_L_MM=${BRAGG_L_MM:-15}
BRAGG_A_0_PM=${BRAGG_A_0_PM:-564.02}
BRAGG_MAX_DISTANCE_PERCENTAGE=${BRAGG_MAX_DISTANCE_PERCENTAGE:-100}
LAU_INPUT=${LAU_INPUT:-data/dummy.csv}
LAU_THRESHOLD=${LAU_THRESHOLD:-0.05}
LAU_DISTANCE=${LAU_DISTANCE:-5}
LAU_PROMINENCE=${LAU_PROMINENCE:-0.05}
LAU_WINDOW=${LAU_WINDOW:-20}

GRAPHS_DIR="$OUT_DIR/graphs"
BRAGG_DIR="$GRAPHS_DIR/bragg"
LAU_DIR="$GRAPHS_DIR/lau"

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

resolve_uv_cmd() {
  if command_exists uv; then
    UV_CMD=(uv)
  elif [ -x ".venv/Scripts/uv.exe" ]; then
    UV_CMD=(".venv/Scripts/uv.exe")
  elif [ -x ".venv/bin/uv" ]; then
    UV_CMD=(".venv/bin/uv")
  else
    UV_CMD=()
  fi
}

resolve_python_cmd() {
  if [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON_CMD=(".venv/Scripts/python.exe")
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_CMD=(".venv/bin/python")
  elif command_exists python3; then
    PYTHON_CMD=(python3)
  elif command_exists python; then
    PYTHON_CMD=(python)
  elif command_exists py; then
    PYTHON_CMD=(py -3)
  else
    PYTHON_CMD=(python)
  fi
}

resolve_uv_cmd
resolve_python_cmd

with_src_pythonpath() {
  if [ -n "${UV_CMD[*]}" ]; then
    "$@"
  else
    if [ -n "${PYTHONPATH:-}" ]; then
      PYTHONPATH="src:$PYTHONPATH" "$@"
    else
      PYTHONPATH="src" "$@"
    fi
  fi
}

run_bragg_cli() {
  if [ -n "${UV_CMD[*]}" ]; then
    "${UV_CMD[@]}" run bragg "$@"
  else
    with_src_pythonpath "${PYTHON_CMD[@]}" -m xray bragg "$@"
  fi
}

run_lau_cli() {
  if [ -n "${UV_CMD[*]}" ]; then
    "${UV_CMD[@]}" run lau "$@"
  else
    with_src_pythonpath "${PYTHON_CMD[@]}" -m xray lau "$@"
  fi
}

case "$MODE" in
  bragg|BRAGG)
    TARGETS=("bragg") ;;
  lau|LAU)
    TARGETS=("lau") ;;
  both|BOTH|all|ALL)
    TARGETS=("bragg" "lau") ;;
  *)
    echo "Unknown mode '$MODE'. Expected bragg, lau, or both." >&2
    exit 1 ;;
esac

for target in "${TARGETS[@]}"; do
  case "$target" in
    bragg)
      mkdir -p "$BRAGG_DIR"
      echo "Generating Bragg graphs into $BRAGG_DIR"
      if [ -f "$BRAGG_IMAGE" ]; then
        if ! run_bragg_cli \
          --output "$BRAGG_DIR" \
          --image "$BRAGG_IMAGE" \
          --big-circle-thresh "$BRAGG_BIG_CIRCLE_THRESH" \
          --small-dot-thresh "$BRAGG_SMALL_DOT_THRESH" \
          --small-dot-thresh-outer "$BRAGG_SMALL_DOT_THRESH_OUTER" \
          --min-spot-area "$BRAGG_MIN_SPOT_AREA" \
          --min-circularity "$BRAGG_MIN_CIRCULARITY" \
          --phys-y-mm "$BRAGG_PHYS_Y_MM" \
          --phys-x-mm "$BRAGG_PHYS_X_MM" \
          --l-mm "$BRAGG_L_MM" \
          --a-0-pm "$BRAGG_A_0_PM" \
          --max-distance-percentage "$BRAGG_MAX_DISTANCE_PERCENTAGE"; then
          echo "Warning: Bragg graph generation failed"
        fi
      else
        echo "Warning: Bragg image '$BRAGG_IMAGE' missing; skipping Bragg gallery"
      fi
      ;;
    lau)
      mkdir -p "$LAU_DIR"
      echo "Generating Lau graphs into $LAU_DIR"
      if [ -f "$LAU_INPUT" ]; then
        if ! run_lau_cli \
          --input "$LAU_INPUT" \
          --output "$LAU_DIR" \
          --threshold "$LAU_THRESHOLD" \
          --distance "$LAU_DISTANCE" \
          --prominence "$LAU_PROMINENCE" \
          --window "$LAU_WINDOW"; then
          echo "Warning: Lau graph generation failed"
        fi
      else
        echo "Warning: Lau input '$LAU_INPUT' missing; skipping Lau gallery"
      fi
      ;;
  esac
done

# Build gallery entries
mkdir -p "$GRAPHS_DIR"
tmp_gallery="$GRAPHS_DIR/.gallery.html"
: > "$tmp_gallery"
shopt -s nullglob
content_written=0
for section in bragg lau; do
  section_dir="$GRAPHS_DIR/$section"
  [ -d "$section_dir" ] || continue
  if [ "$(ls -A "$section_dir")" ]; then
    echo "<h2>${section^} Graphs</h2>" >> "$tmp_gallery"
    content_written=1
    for f in "$section_dir"/*; do
      [ -f "$f" ] || continue
      name=$(basename "$f")
      rel_path="$section/$name"
      ext=${name##*.}
      case "${ext,,}" in
        png|jpg|jpeg|gif|svg|webp)
          echo "<figure><a href=\"$rel_path\"><img src=\"$rel_path\" alt=\"$name\"></a><figcaption>$name</figcaption></figure>" >> "$tmp_gallery" ;;
        html)
          echo "<p><a href=\"$rel_path\">$name</a></p>" >> "$tmp_gallery" ;;
        *)
          echo "<p><a href=\"$rel_path\">$name</a> (${ext})</p>" >> "$tmp_gallery" ;;
      esac
    done
  fi
done

if [ "$content_written" -eq 0 ]; then
  echo "<p>No graph outputs were generated in this run.</p>" >> "$tmp_gallery"
fi

# Render template by injecting the gallery into {{GALLERY}}
awk '{
  if ($0 ~ /\{\{GALLERY\}\}/) {
    while ((getline line < ARGV[1]) > 0) print line; next
  } print
}' "$tmp_gallery" "$TEMPLATE" > "$GRAPHS_DIR/index.html"
