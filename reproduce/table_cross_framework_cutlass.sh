#!/usr/bin/env bash
# Paper tab:cross-framework — CUTLASS rows (ncu-measured on hardware).
#
# Builds the CUTLASS SM90 cooperative warp-specialized GEMM twice:
#   baseline — pristine CUTLASS v4.5.0 headers (struct: no overlap)
#   SALA     — v4.5.0 headers + the SALA struct->union patch
#              (benchmarks/cutlass/patches/sala_union.patch)
# then ncu-measures launch__shared_mem_per_block_dynamic for both.
#
# The patch is the manual workaround the paper describes (struct->union): the
# mainloop and epilogue tensor storages become a union (max instead of sum).
# It adds no synchronization of its own -- README section 2.3 states when the
# overlap is valid (one work tile per CTA) and why the persistent multi-tile
# assignment would additionally need producer-side gating.
#
# Step 4 verifies the numerics of both builds (sampled fp32 reference) and
# exits nonzero on mismatch.
#
# Usage: GPU=0 bash reproduce/table_cross_framework_cutlass.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CUTLASS="$REPO/benchmarks/cutlass"
SRC="$CUTLASS/src/cutlass_union_test.cu"
PATCH="$CUTLASS/patches/sala_union.patch"
BASE_INC="${CUTLASS_HOME:-$CUTLASS/extern/cutlass_include}"  # pristine v4.5.0 repo root
UTIL_INC="$BASE_INC/tools/util/include"                      # cutlass/util helpers
WORK="${WORK:-/tmp/ae_cutlass}"
GPU="${GPU:-0}"

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found"; exit 1; }
command -v ncu >/dev/null || { echo "ERROR: ncu not found"; exit 1; }
[[ -f "$SRC" && -d "$BASE_INC/include" && -d "$UTIL_INC" && -f "$PATCH" ]] || {
    echo "ERROR: missing sources under $CUTLASS"; exit 1; }

NCU_METRIC="launch__shared_mem_per_block_dynamic"
CONFIGS=("128x128 2s" "128x128 3s" "128x128 4s" "128x256 2s" "128x256 3s")

mkdir -p "$WORK"

echo "================================================================"
echo " Cross-framework SMEM — CUTLASS rows (tab:cross-framework)"
echo " Cooperative TMA warp-specialized GEMM, f16, CUTLASS v4.5.0"
echo " Baseline: struct  |  SALA: struct->union (patched header)"
echo " GPU: $GPU"
echo "================================================================"

# ---- 1. Baseline binary (pristine v4.5.0 headers) ----
echo "[1/4] Building baseline (struct) ..."
nvcc -std=c++17 -arch=sm_90a -O2 \
    -I "$BASE_INC/include" -I "$UTIL_INC" \
    "$SRC" -o "$WORK/cutlass_union_test_baseline"

# ---- 2. SALA binary (patched header copy) ----
echo "[2/4] Building SALA (struct->union) ..."
rm -rf "$WORK/cutlass_sala_include"
mkdir -p "$WORK/cutlass_sala_include"
cp -r "$BASE_INC/." "$WORK/cutlass_sala_include/"
(cd "$WORK/cutlass_sala_include/include" && patch -p1 --forward -s < "$PATCH")
nvcc -std=c++17 -arch=sm_90a -O2 \
    -I "$WORK/cutlass_sala_include/include" -I "$UTIL_INC" \
    "$SRC" -o "$WORK/cutlass_union_test_sala"

# ---- 3. ncu both binaries (5 kernels each, in CONFIGS order) ----
echo "[3/4] ncu measuring ..."
mapfile -t base_smem < <(CUDA_VISIBLE_DEVICES=$GPU ncu --metrics $NCU_METRIC \
    "$WORK/cutlass_union_test_baseline" 2>&1 \
    | grep "$NCU_METRIC" | grep -oP '[0-9]+\.[0-9]+')
mapfile -t sala_smem < <(CUDA_VISIBLE_DEVICES=$GPU ncu --metrics $NCU_METRIC \
    "$WORK/cutlass_union_test_sala" 2>&1 \
    | grep "$NCU_METRIC" | grep -oP '[0-9]+\.[0-9]+')

[[ ${#base_smem[@]} -eq 5 && ${#sala_smem[@]} -eq 5 ]] || {
    echo "ERROR: expected 5 kernels per binary, got ${#base_smem[@]}/${#sala_smem[@]}"; exit 1; }

echo ""
printf "%-12s  %-10s  %-10s  %-6s  %s\n" "Config" "Base KB" "SALA KB" "Save%" "Paper"
printf "%-12s  %-10s  %-10s  %-6s  %s\n" "------" "-------" "-------" "-----" "-----"
for i in "${!CONFIGS[@]}"; do
    b="${base_smem[$i]}"; s="${sala_smem[$i]}"
    sp=$(echo "scale=1; 100*($b-$s)/$b" | bc)
    case "$i" in
        0) paper="100 -> 67 (34%)" ;;
        1) paper="133 -> 99 (25%)" ;;
        2) paper="166 -> 132 (20%)" ;;
        3) paper="133 -> 99 (25%)" ;;
        4) paper="(not in Table 2)" ;;
    esac
    printf "%-12s  %-10s  %-10s  %-6s  %s\n" "${CONFIGS[$i]}" "$b" "$s" "$sp%" "$paper"
done

echo ""
echo "Kernel order matches cutlass_union_test.cu main(): 5 analyze_and_run calls."
echo "Paper values are these ncu measurements rounded to whole KB."

# ---- 4. Numerical verification (sampled fp32 reference) ----
# The baseline is verified at the measurement size (2048^3).  The union build
# shares the epilogue staging with the mainloop stages, which is only safe when
# a CTA owns a single work tile -- the persistent 2048^3 assignment needs
# cross-tile producer gating that this manual workaround does not implement
# (see the README section 2.3).  So the union is verified at 1024^2, where
# every CTA gets one work tile, and the 2048^3 union run reports NOT CHECKED.
echo ""
echo "[4/5] checker self-test (negative tests: NaN / Inf / large error must be rejected)"
"$WORK/cutlass_union_test_baseline" --selftest || { echo "ERROR: the reference checker failed its own negative tests"; exit 1; }

echo ""
echo "[5/5] numerical verification (sampled fp32 reference, 4096 samples):"
echo "--- baseline (struct) ---"
"$WORK/cutlass_union_test_baseline" --check 2>&1 | grep -E "Reference|Result|Done|Non-persistent|overlapped|checked:|skipped:"
b_rc=${PIPESTATUS[0]}
echo "--- SALA (struct->union) ---"
"$WORK/cutlass_union_test_sala" --check 2>&1 | grep -E "Reference|Result|Done|Non-persistent|overlapped|checked:|skipped:"
s_rc=${PIPESTATUS[0]}
if [[ $b_rc -ne 0 || $s_rc -ne 0 ]]; then
    echo "ERROR: numerical verification failed (baseline rc=$b_rc, union rc=$s_rc)"
    exit 1
fi
echo "Both builds numerically verified (self-test included)."
echo "Scope: baseline 5/5 at 2048^3 + 3/3 non-persistent; union 5/5 in the single-tile"
echo "regime (1024^2) + 3/3 non-persistent. The persistent union at 2048^3 is NOT"
echo "checked by design -- multi-tile assignment without producer-side gating is not"
echo "a valid overlap (README 2.3); those rows are SMEM measurements, not timings."
