#!/usr/bin/env bash
# Paper tab:cross-framework — Tawa rows (ncu-measured on hardware).
#
# Tawa = the vendored triton-aref fork (benchmarks/tawa/triton-aref, committed
# directly in the repo), `aref_auto_ws`
# branch (Triton v3.3.0 fork) with the SALA commits 3777fb9 (allocator
# post-pass) + ce1ee01 (scf_to_cf pass-order fix).
#   baseline — SALA_ENABLE=0 (upstream allocator)
#   SALA     — SALA_ENABLE=1 (SALA allocator post-pass in lib/Analysis/Allocation.cpp)
# ncu-measures launch__shared_mem_per_block_dynamic in both modes.
#
# Kernels (benchmarks/tawa/, vendored from the original evaluation):
#   tawa_sala_config_test.py — persistent nested warp-specialized GEMM
#     (grid = NUM_SMS CTAs; kernel `matmul_kernel_nested`)
#   tawa_fmha_ncu.py         — TMA FMHA from test_tma_fused_attention.py,
#     WG_SPEC="mma_first", MATH_WG_PIPE=True (kernel `_attn_fwd`)
#
# CRITICAL: each mode must compile with a FRESH Triton JIT cache dir. A kernel
# cached from a compile in the other mode would be reused unchanged (the SALA
# env var does not change the cache key), silently hiding the SALA effect.
#
# FMHA 3s baseline cannot launch: 264 KB > 227 KB hardware limit, Triton raises
# OutOfResources("Required: 264344 ..."). The script then reports the
# compiler-requested size with `*` — this is exactly the paper's `*` meaning
# (compiler-reported) on the "FA WS 3s 258* -> 230" row.
#
# After the SMEM rows the script verifies numerics and exits nonzero on
# mismatch: the GEMM rows against a fp32 torch matmul reference (rel. err
# < 5 %), the FMHA rows against a fp32 causal-attention reference.  FMHA runs
# with --membar 1 (the cross-tile mbarrier the SALA overlap requires; without
# it the 3-stage kernel races -- see README section 2.4).  The 3-stage
# baseline cannot launch, so only its SALA side is checked.
#
# Usage: GPU=0 bash reproduce/table_cross_framework_tawa.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAWA="$REPO/benchmarks/tawa"
GEMM="$TAWA/tawa_sala_config_test.py"
FMHA="$TAWA/tawa_fmha_ncu.py"
PY="${PYTHON:-python3.10}"
GPU="${GPU:-0}"

command -v ncu >/dev/null || { echo "ERROR: ncu not found"; exit 1; }
command -v bc >/dev/null || { echo "ERROR: bc not found"; exit 1; }
command -v "$PY" >/dev/null || { echo "ERROR: $PY not found (set PYTHON=...)"; exit 1; }

# The vendored fork (benchmarks/tawa/triton-aref) is used in two ways:
#   - its source tree provides the SALA allocator pass (checked below);
#   - the fork itself must be installed (`cd triton-aref && pip install .`) —
#     the backends (nvidia/amd) live in third_party/ and are packaged only
#     into the installed wheel, so the kernels import the INSTALLED triton
#     (3.3.0), not the source tree via PYTHONPATH.
if [[ -z "${TRITON_AREF:-}" ]]; then
    for d in "$TAWA/triton-aref"; do
        [[ -d "$d/python" ]] && { TRITON_AREF="$d"; break; }
    done
fi
[[ -d "$TRITON_AREF/python" ]] || {
    echo "ERROR: triton-aref not found. The fork is vendored in the repository"
    echo "       (benchmarks/tawa/triton-aref); re-clone the repo if missing."
    echo "       (or set TRITON_AREF=/path/to/triton-aref)"; exit 1; }
"$PY" -c "import triton" >/dev/null 2>&1 || {
    echo "ERROR: triton cannot be imported. Build the fork:"
    echo "       cd $TRITON_AREF && pip install .   # builds the fork"; exit 1; }
"$PY" -c "import triton; assert triton.__version__.split('+')[0] == '3.3.0'" 2>/dev/null || {
    echo "ERROR: installed triton is not the aref fork (3.3.0)."
    echo "       cd $TRITON_AREF && pip install .   # builds the fork"; exit 1; }
grep -q 'SALA_ENABLE' "$TRITON_AREF/lib/Analysis/Allocation.cpp" || {
    echo "ERROR: $TRITON_AREF lacks the SALA allocator pass (wrong branch/commit?)"
    echo "       expected aref_auto_ws branch (SALA commits 3777fb9 + ce1ee01),"
    echo "       lib/Analysis/Allocation.cpp gate"; exit 1; }

# the FMHA kernel resolves its helpers (fmha_common, test_tma_fused_attention)
# from $TRITON_AREF/python/test/unit/auto_ws
export TRITON_AREF
NCU_METRIC="launch__shared_mem_per_block_dynamic"

# smem_of <ncu-out> <kernel>: ncu-measured Kbyte of the first launch of <kernel>
smem_of() {
    awk -v k="$2" '$0 ~ "  " k " \\(" {f=1; next} f && /Kbyte/ {print $NF; exit}' "$1"
}
# required_of <ncu-out>: compiler-requested bytes from Triton's OutOfResources
required_of() {
    grep -oE "Required: [0-9]+" "$1" | grep -oE "[0-9]+" | head -1
}

# measure <label> <kernel> <kind> <script> <args...> -> sets BASE_KB, SALA_KB, BASE_STAR
BASE_KB=; SALA_KB=; BASE_STAR=
measure() {
    local label=$1 kern=$2 kind=$3 script=$4; shift 4
    local bf sf
    bf=$(mktemp); sf=$(mktemp)
    rm -rf /tmp/ae_tawa_cache_b /tmp/ae_tawa_cache_s
    mkdir -p /tmp/ae_tawa_cache_b /tmp/ae_tawa_cache_s
    if [[ "$kind" == gemm ]]; then
        CUDA_VISIBLE_DEVICES=$GPU SALA_ENABLE=0 TRITON_CACHE_DIR=/tmp/ae_tawa_cache_b \
            ncu --metrics $NCU_METRIC "$PY" "$script" "$@" > "$bf" 2>&1 || true
        CUDA_VISIBLE_DEVICES=$GPU SALA_ENABLE=1 TRITON_CACHE_DIR=/tmp/ae_tawa_cache_s \
            ncu --metrics $NCU_METRIC "$PY" "$script" "$@" > "$sf" 2>&1 || true
    else
        SALA_ENABLE=0 TRITON_CACHE_DIR=/tmp/ae_tawa_cache_b \
            ncu --metrics $NCU_METRIC "$PY" "$script" --gpu "$GPU" "$@" > "$bf" 2>&1 || true
        SALA_ENABLE=1 TRITON_CACHE_DIR=/tmp/ae_tawa_cache_s \
            ncu --metrics $NCU_METRIC "$PY" "$script" --gpu "$GPU" "$@" > "$sf" 2>&1 || true
    fi
    BASE_KB=$(smem_of "$bf" "$kern")
    if [[ -z "$BASE_KB" ]]; then                 # kernel did not launch (OOR)
        local req; req=$(required_of "$bf")
        BASE_KB=$(echo "scale=2; $req/1024" | bc)
        BASE_STAR="*"
    fi
    SALA_KB=$(smem_of "$sf" "$kern")
    [[ -n "$BASE_KB" && -n "$SALA_KB" ]] || {
        echo "ERROR: could not extract SMEM for kernel '$kern' ($script $*)"; exit 1; }
}

echo "================================================================"
echo " Cross-framework SMEM — Tawa rows (tab:cross-framework)"
echo " triton-aref aref_auto_ws branch (Triton v3.3.0), NVWS kernels"
echo " Baseline: SALA_ENABLE=0  |  SALA: SALA_ENABLE=1"
echo " GPU: $GPU (GEMM via CUDA_VISIBLE_DEVICES, FMHA via --gpu)"
echo "================================================================"

rows=(
    "128x128 3s|131 -> 98 (25%)|matmul_kernel_nested|gemm|$GEMM|--bm|128|--bn|128|--bk|64|--stages|3"
    "128x128 2s|98 -> 66 (33%)|matmul_kernel_nested|gemm|$GEMM|--bm|128|--bn|128|--bk|64|--stages|2"
    "64x128 2s |66 -> 49 (25%)|matmul_kernel_nested|gemm|$GEMM|--bm|64|--bn|128|--bk|64|--stages|2"
    "FA WS 2s  |199 -> 164 (18%)|_attn_fwd|fmha|$FMHA|--stages|2"
    "FA WS 3s  |258* -> 230 (11%)|_attn_fwd|fmha|$FMHA|--stages|3"
)

echo ""
printf "%-10s  %-10s  %-10s  %-6s  %s\n" "Config" "Base KB" "SALA KB" "Save%" "Paper"
printf "%-10s  %-10s  %-10s  %-6s  %s\n" "------" "-------" "-------" "-----" "-----"
for r in "${rows[@]}"; do
    IFS='|' read -r label paper kern kind script rest <<< "$r"
    IFS='|' read -r -a args <<< "$rest"
    measure "$label" "$kern" "$kind" "$script" "${args[@]}"
    save=$(echo "scale=1; 100*($BASE_KB-$SALA_KB)/$BASE_KB" | bc)
    printf "%-10s  %-10s  %-10s  %-6s  %s\n" \
        "$label" "$BASE_KB$BASE_STAR" "$SALA_KB" "$save%" "$paper"
done

echo ""
echo "* = compiler-reported (kernel cannot launch: 264 KB > 227 KB HW limit);"
echo "   matches the paper's * on \"Tawa FA WS 3s 258* -> 230\"."
echo "Paper values are these ncu measurements rounded to whole KB."

# ---- Numerical verification of the FMHA rows (outside ncu) ----
# The SMEM rows above only measure allocation; the FMHA kernel is the one row
# whose output can be checked against a reference without leaving the fork's
# stack, so we do: fp32 torch causal-attention reference, 2% + 2% tolerance.
# (The GEMM rows run the fork's own kernels with the SALA allocator gate; they
# are SMEM measurements too and are not numerics-checked here.)
echo ""
echo "GEMM numerical verification (fp32 torch matmul reference, 4096^3):"
gemm_rc=0
for cfg in "128 128 64 3" "128 128 64 2" "64 128 64 2"; do
    set -- $cfg
    for mode in 0 1; do
        # fresh JIT cache per mode: SALA_ENABLE is not part of Triton's cache key
        rm -rf /tmp/ae_tawa_check_cache && mkdir -p /tmp/ae_tawa_check_cache
        out=$(SALA_ENABLE=$mode TRITON_CACHE_DIR=/tmp/ae_tawa_check_cache \
              CUDA_VISIBLE_DEVICES=$GPU "$PY" "$GEMM" --bm "$1" --bn "$2" --bk "$3" --stages "$4" 2>&1) \
              || gemm_rc=1
        echo "  ${1}x${2} ${4}s SALA_ENABLE=$mode: $(echo "$out" | grep -E 'Correctness' | tail -1)"
    done
done
if [[ $gemm_rc -ne 0 ]]; then
    echo "ERROR: GEMM numerical verification failed"
    exit 1
fi

echo ""
echo "FMHA numerical verification (fp32 torch reference, causal, SEQ=4096):"
fmha_rc=0
for st in 2 3; do
    for mode in 0 1; do
        if [[ $st -eq 3 && $mode -eq 0 ]]; then
            echo "  SALA_ENABLE=0 stages=3: skipped (baseline exceeds the 227 KB HW"
            echo "    limit and cannot launch - the paper's star, SMEM is compiler-reported)"
            continue
        fi
        # fresh JIT cache per mode: SALA_ENABLE is not part of Triton's cache key
        rm -rf /tmp/ae_tawa_check_cache && mkdir -p /tmp/ae_tawa_check_cache
        out=$(SALA_ENABLE=$mode TRITON_CACHE_DIR=/tmp/ae_tawa_check_cache \
              CUDA_VISIBLE_DEVICES=$GPU "$PY" "$FMHA" --stages "$st" --gpu "$GPU" --check 2>&1) || fmha_rc=1
        echo "  SALA_ENABLE=$mode stages=$st: $(echo "$out" | grep -E 'FMHA reference' | tail -1)"
    done
done
if [[ $fmha_rc -ne 0 ]]; then
    echo "ERROR: FMHA numerical verification failed"
    exit 1
fi
echo "FMHA outputs match the reference."
