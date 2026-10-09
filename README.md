# Artifact Evaluation — Signal-Aware Liveness Analysis for Shared Memory Optimization in Warp-Specialized GPU Kernels (CGO 2027)

This repository contains the complete artifact for the paper: the
compiler, the kernels, and one-command reproduction scripts for every
XComp statistic in Figure 2, Table 2 (cross-framework), Table 3
(occupancy), Figure 3 (throughput), and Table 4 (ablation). A fresh
clone on an H100/H800 reproduces all compiler-reported and
`ncu`-measured shared-memory values exactly, plus register counts and
occupancy. Every XComp kernel passes its CPU-reference correctness
check in both modes; the Tawa rows are checked against fp32 torch
references (GEMM matmul; FMHA causal attention) and the CUTLASS rows
against a sampled fp32 reference — see §2.3/§2.4 for the verification
and §6.3 for the one documented limit (the CUTLASS union under
persistent multi-tile scheduling).

| Requirement | Value |
|---|---|
| GPU | NVIDIA H100 or H800 (SM90a); the paper's measurements are from an H800 |
| CUDA | 13.0 (the container's base; reproduces the paper's registers exactly — see §6.1 for newer minors) |
| Tools | `cmake` ≥ 3.18, `ninja`, `flex`, `bison`, `gcc`/`g++`, `git`, `python3`, `bc`, `ncu` (ships with the CUDA toolkit) |
| Network | none for the compiler build (CUTLASS bundled); the Tawa fork's LLVM build ~1.2 GB at its one-time setup |
| Time | build ~10 min; Figure 2 ~10 s (no GPU); Table 2 ~30 min (GPU + ncu) |

## 0. Build the Compiler

```bash
git clone --recursive https://github.com/LancerLab/sala-artifacts.git && cd sala-artifacts
```

The compiler is the `croqtile/` submodule, pinned to the paper-lineage
branch's commit `a9cd1ba` (the compiler state the paper was evaluated with). All commands in this guide are run from
the **repository root**.

```bash
cmake -S croqtile -B croqtile/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCHOREO_DEFAULT_TARGET=cute
ninja -C croqtile/build choreo copp
```

CUTLASS v4.2.1 is fetched at configure only if
`croqtile/extern/cutlass` is missing — the artifact bundles it, so the
build needs no network. An explicit `-DCUTE_HOME=` or `CUTE_HOME`
always takes precedence. Verify:

```bash
croqtile/build/choreo --help-hidden | grep no-sala     # shows: --no-sala
```

All reproduction scripts resolve the compiler themselves
(`$REPO/croqtile/build/choreo`); override with `CHOREO=/path/to/choreo`.

---

## 1. Figure 2 (fig:hb-results) — Compiler-Reported SMEM Allocation (~10 s, no GPU)

These are the compiler-internal allocation sizes behind the paper's
Figure 2 ("SALA liveness refinement results (compiler allocation)") —
the direct effect of SALA's tighter liveness intervals, before `nvcc`
adds ~4–20 KB of runtime metadata.

### 1.1 One command

```bash
bash reproduce/figure2_compiler_smem.sh
```

Expected output — all nine items:

```
Fig.2 item      Base KB   SALA KB   Save%   Paper
----------      -------   -------   -----   ------------
1P1C f16        112 KB    96 KB     14.2%   112->96 (-14%)
1P1C e4m3       112 KB    96 KB     14.2%   112->96 (-14%)
1P2C f16        128 KB    96 KB     25.0%   128->96 (-25%)
1P3C f16        128 KB    80 KB     37.5%   128->80 (-37%)
FA K/V          40 KB     32 KB     20.0%   40->32 (-20%)
FA full         72 KB     56 KB     22.2%   72->56 (-22%)
FA tuned        192 KB    160 KB    16.6%   192->160 (-17%)
FA 3s           256 KB*   224 KB    12.5%   256*->224 (-12%)
Conv2d          96 KB     64 KB     33.3%   96->64 (-33%)
```

### 1.2 Kernel-compiled items (1–4, 7–9)

Per-kernel command (choreo prints the dynamic SMEM as
`cudaFuncAttributeMaxDynamicSharedMemorySize is set to \`N`):

```bash
croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala <kernel.co> -o /dev/null 2>&1 | grep -o 'set to `[0-9]*'
croqtile/build/choreo -gs -t cute -arch=sm_90a           <kernel.co> -o /dev/null 2>&1 | grep -o 'set to `[0-9]*'
```

| Item | Kernel file | Base B | SALA B | Paper |
|------|-------------|--------|--------|-------|
| 1P1C f16 | `benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co` | 114688 | 98304 | 112→96 |
| 1P1C e4m3 | `benchmarks/matmul/matmul_e4m3_dyn_sm90_warpspec_1p1c.co` | 114688 | 98304 | 112→96 |
| 1P2C f16 | `benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p2c.co` | 131072 | 98304 | 128→96 |
| 1P3C f16 | `benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` | 131072 | 81920 | 128→80 |
| FA tuned | `benchmarks/flash_atten/fa_tuned_1p2c.co` | 196608 | 163840 | 192→160 |
| FA 3s | `benchmarks/flash_atten/fa_3s.co` | 262144 (cannot compile) | 229376 | 256*→224 |
| Conv2d | `benchmarks/conv/conv2d_fprop_ws_1p1c.co` | 98304 | 65536 | 96→64 (−33%) |

The FA 3s baseline (item 8) cannot compile — it exceeds the 228 KB/SM
limit ("shared memory OUT OF BOUND"); 256 KB is the kernel's buffer
sum. SALA overlaps the 32 KB output buffer with the K/V pipeline →
224 KB. This is the paper's `256*` star.

### 1.3 HB-analyzer model items (5, 6)

Items 5 and 6 are HB-pattern models from the vendored analyzer
(`hb_analyzer/`, stdlib-only Python) — the pattern JSONs at
`hb_analyzer/examples/choreo_fa_kv.json` and
`choreo_fa_full.json`:

```bash
PYTHONPATH=. python3 -m hb_analyzer all-examples
```

prints (among others): `fa_kv_overlap` 40960→32768 (40→32),
`fa_fwd_full_pipeline` 73728→57344 (72→56).
Item 9 (Conv2d) is a real kernel compile with a CPU-reference
correctness check (Test Passed in both modes) — the paper's
"verified by compilation and correctness testing".

### 1.4 Verification (correctness)

Every **executed** XComp kernel (items 1–4, 7–9) passes its harness's
independent CPU-reference check in both modes (`--no-sala` and SALA).
Items 5 and 6 are HB-analyzer *models* (`hb_analyzer/`, a standalone
stdlib-only Python tool) — they are static-analysis results, not
executed kernels, and have no runtime output to check:

| Items | Kind | Check | Result |
|-------|------|-------|--------|
| GEMM (f16 1P1C/1P2C/1P3C, e4m3) | kernel | harness CPU reference, 2048³ (fp32 full-K dot, 5% tolerance) | **Test Passed** ×2 modes each |
| FA tuned 1P2C | kernel | naive-attention reference (0 failed samples required), B=2 H=16 SEQ=512..16384 | **Test Passed**, fail_rate=0, ×2 modes |
| FA 3s | kernel | same harness, SALA side | **Test Passed** (baseline cannot launch) |
| FA 1P1C 2s | kernel | built-in naive-attention reference, 5 configs (SEQ 512–8192) | **Test Passed** 5/5, ×2 modes |
| Conv2d (item 9) | kernel | row-row sampled GEMM check (fp32 reference, 5% tolerance) | **Test Passed** ×2 modes |
| HB-analyzer items 5, 6 | model | n/a — static analysis over vendored pattern JSONs | n/a |

`Test Passed` (or the scripts' `[Test Passed]` per row) is the success
marker; `Test FAILED`, a nonzero `fail_rate`, or `[NO-RUN]` (no
verdict within the run window) means the run did not verify — re-run
it before concluding.

---

## 2. Table 2 (tab:cross-framework) — ncu Hardware Validation (~30 min, needs GPU+ncu)

The paper's table has three framework parts — XComp, CUTLASS, Tawa.
All rows are `ncu`-measured (`launch__shared_mem_per_block_dynamic`)
except the starred baselines (which cannot launch).

```bash
GPU=0 bash reproduce/run_all.sh table2    # the whole Table 2, one command
# per-family pieces:
GPU=0 bash reproduce/table3_xcomp.sh               # XComp GEMM rows (~5 min)
GPU=0 bash reproduce/table2_xcomp_fa.sh            # XComp FA rows (~10 min)
GPU=0 bash reproduce/table_cross_framework_cutlass.sh   # CUTLASS rows
GPU=0 bash reproduce/table_cross_framework_tawa.sh      # Tawa rows
```

### 2.1 XComp — GEMM rows

| Row | Base KB (ncu) | SALA KB (ncu) | Paper |
|---|---|---|---|
| 1P1C 64x128 4s | 115.7 | 99.3 | 116→99 (15%) |
| 1P1C e4m3 (64x128 4s) | 115.7 | 99.3 | 116→99 (15%) |
| 1P1C 64x128 3s | 91.1 | 74.8 | 91→75 (18%) |
| 1P2C GEMM | 132.1 | 99.3 | 132→99 (25%) |
| 1P3C GEMM | 132.1 | 82.9 | 132→83 (37%) |

The script also prints occupancy limits, register counts, and
active-warp percentages (see §3).

### 2.2 XComp — FA rows (incl. correctness)

| Row | Base KB (ncu) | SALA KB (ncu) | Paper | Correctness |
|---|---|---|---|---|
| FA 1P1C 2s | 82.05 | 73.86 | 82→74 (10%) | Test Passed |
| FA tuned 1P2C | 196.74 | 163.97 | 197→164 (17%) | Test Passed |
| FA 3s | cannot launch | 229.50 | 256*→230 | Test Passed |

(All three FA rows are the paper's current values — the FA 1P1C 2s row
was updated at camera-ready to the measured ncu pair, 82→74.
The script also prints the ncu Act. at SEQ=16384, matching Table 3's
FA rows; see §3.)

### 2.3 CUTLASS rows (own toolchain, ~15 min incl. setup)

```bash
bash setup_cutlass.sh   # one-time: v4.5.0 headers + SALA struct->union patch
GPU=0 bash reproduce/table_cross_framework_cutlass.sh
```

| Config | Baseline | SALA (union) | Paper |
|--------|----------|-------------|-------|
| Coop 128² 2s | 100.35 | 66.56 | 100→67 (34%) |
| Coop 128² 3s | 133.12 | 99.33 | 133→99 (25%) |
| Coop 128² 4s | 165.89 | 132.10 | 166→132 (20%) |
| Coop 128×256 2s | 133.12 | 99.33 | 133→99 (25%) |

**Numerical verification and its limit.** The script verifies both
binaries against a sampled fp32 reference (4096 output elements,
fp32 dot products over the same fp16 inputs; tolerance
`0.1 + 1% · |ref|`), and exits nonzero on any mismatch:

- the **baseline** (pristine v4.5.0 struct) is verified at the
  measurement size (2048³) — max |D−D_ref| ≈ 0.007;
- the **union** build is verified at 1024² per side, where every CTA
  owns a single work tile — max |D−D_ref| ≈ 0.008. At the 2048³
  measurement size the union run reports
  `Reference: NOT CHECKED here` for the reason below.

*Why the union is only verified in that regime:* this is a question of
**validity**, not of speed. SALA's analysis answers one question — do the
signals and synchronization points permit the two buffers' live ranges to
be tightened (are the phases disjoint)? If yes, the overlap is sound and
strictly better (less SMEM, same work); if no, SALA does nothing.

For the persistent cooperative kernel the answer is **no**: a CTA owns
several work tiles (>114 tiles on a 114-SM H800), so tile *T*'s epilogue
and tile *T+1*'s mainloop are live at the same time with no synchronization
point between them — the next tile's mainloop TMA loads overwrite the
shared memory while the previous tile's epilogue is still staging its
stores. A `NamedBarrier::sync` between `mma_tail()` and `epilogue.store()`
synchronizes the consumer warps only and does not create that ordering; we
measured it: with the barrier in place the union still mismatches ~2,300 of
4,096 sampled elements at 2048³. The 1024² run (one work tile per CTA) *is*
the valid regime — there the union is exact (max |D−D_ref| ≈ 0.008), and
unions of exactly this kind are what CUTLASS itself ships for its
non-persistent kernel (`sm90_gemm_tma_warpspecialized.hpp`: *"Mainloop and
epilogue don't use smem concurrently since kernel is non-persistent, so we
can use a union"*).

That gives a cross-check of the analysis against a production framework's
hand decisions: of CUTLASS's 21 warp-specialized kernel files, the 2 that
overlap their mainloop and epilogue storage are exactly the non-persistent
ones (valid by SALA's criterion), and the 19 that keep `struct` are exactly
the persistent ones (invalid without a cross-tile barrier). The script
demonstrates the valid case directly: `--check` also runs CUTLASS's
non-persistent kernel (its own union, unpatched) and verifies its numerics.

So the Table-2 CUTLASS rows measure the *size of the opportunity* in that
kernel; whether the overlap may be taken is SALA's verdict, and for the
persistent kernel as shipped the verdict is no. The valid case is
demonstrated with numbers: the non-persistent kernel's storage is 96.0 KB
(mainloop) + 17.0 KB (epilogue) = **97.0 KB** overlapped instead of 113.0 KB
laid out separately, and its output matches the reference (max
|D−D_ref| = 0.007, 0/4096 bad) at the full 2048³ size — verified for three
configurations in both builds.

### 2.4 Tawa rows (own toolchain, Python 3.10, ~30 min incl. setup)

```bash
pip install -r benchmarks/tawa/requirements.txt            # one-time (~2.5 GB)
cd benchmarks/tawa/triton-aref && pip install . && cd -   # one-time
GPU=0 bash reproduce/table_cross_framework_tawa.sh
```

(Network notes: json, googletest, and the NVIDIA redistributables are
bundled in the repo and pre-placed by the Dockerfile, so `pip install .`
downloads only the LLVM build from Microsoft's blob host (~1.2 GB,
cached under `~/.triton` — retry on timeout). pip's "torch requires
triton==3.7.1" conflict warning is expected and harmless — the kernels
use torch only for tensor allocation.)

| Config | Baseline | SALA | Paper |
|--------|----------|------|-------|
| 128² 3s | 131.13 | 98.36 | 131→98 (25%) |
| 128² 2s | 98.34 | 65.57 | 98→66 (33%) |
| 64×128 2s | 65.57 | 49.18 | 66→49 (25%) |
| FA WS 2s | 198.74 | 163.93 | 199→164 (18%) |
| FA WS 3s | 258.1* | 229.53 | 258*→230 (11%) |

**FMHA synchronization (and a checked reference).** The FMHA kernel is
run with `--membar 1` (`tawa_fmha_ncu.py`) — the cross-tile mbarrier
synchronization the SALA overlap requires, and part of the paper's Tawa
prototype ("full SALA prototype, cross-tile `bar.sync`"). **Without it
the 3-stage kernel races**: we measured 192–832 NaN elements per run in
that configuration (varying between runs), while with it the output is
exact. The SMEM numbers are identical either way (163.93 / 229.53 KB),
so the rows reproduce the paper exactly.

The synchronization is not free: on the H800 at SEQ=4096 the FMHA
kernel takes ~1.65 ms with `--membar 1` vs ~1.45 ms without it
(+7–14 %; the "without" build is the racy one, so this is the price of
correctness, not a regression against a valid baseline). No paper claim
depends on Tawa timing — its rows are SMEM/occupancy measurements, both
unchanged by the barrier.

**Prototype scope of the Tawa pass.** The fork's SALA pass is a ~50-LOC
prototype of the interference refinement, not the production analysis
(XComp, §0): it treats aref pipeline buffers and non-aref warp-group
buffers as phase-disjoint and compacts the latter to offset 0. Two limits
matter if you reuse it:

- The cross-tile `bar.sync` it inserts covers the loop shapes it
  recognizes (`WarpGroupOp > for > for`, i.e. the GEMM kernels), and the
  thread count it emits assumes the 2 × 128-thread warp-group split — the
  `num_warps=4` used by the rows above. Any other `num_warps` (2, 8, …)
  would leave the barrier half-reached (the kernel deadlocks), so the
  pass now **rejects an unsupported geometry at compile time** with an
  explicit error rather than emitting a broken barrier — use
  `num_warps=4`, or the production analysis, for other geometries.
- The FMHA kernel's cross-tile synchronization comes from the kernel's
  own mbarrier protocol (`--membar 1`), not from the pass — see above.
  The pass inserts nothing for that loop shape, and the analysis does not
  verify the yield-point barrier; it is a precondition the kernel
  template supplies.

`SALA_NO_BARRIER=1` disables the inserted barrier. It is a diagnostic
switch that deliberately produces unsound kernels (it is how the FMHA
hazard above was demonstrated) — do not use it for measurements.

The supported Tawa path is the two scripts under `benchmarks/tawa/`:
they carry the reference checks and the cross-tile synchronization. The
compiler submodule also ships internal evaluation scripts
(`croqtile/tools/sala_real_eval/`); several of them predate the
cross-tile sync and are **not** suitable for three-stage kernels — use
the scripts above.

The script also checks the FMHA outputs against a **fp32 torch causal
reference** (2 % + 2 % tolerance, `--check`) for both stages and both
modes and exits nonzero on mismatch — max |o − ref| ≈ 0.001. The
3-stage baseline cannot launch (it exceeds the 227 KB limit — the
paper's `*`), so only the SALA side of that row is checked.

---

## 3. Table 3 (tab:occupancy) — Registers, Occupancy, and Act.

The script collects `launch__registers_per_thread`, the occupancy
limits, and the ncu-measured **Act. column**
(`sm__warps_active.avg.pct_of_peak_sustained_active`). On CUDA 13.0
(the paper's environment):

| Row | Regs (no-SALA → SALA) | Paper |
|-----|------------------------|-------|
| 1P1C 64x128 4s | 72 → 72 | 72 ✓ |
| 1P2C GEMM (3s) | 95 → 96 | 95→96 ✓ |
| 1P3C GEMM (2s) | 101 → 96 | 101→96 ✓ |
| FA tuned 1P2C | 168 → 168 | 168 ✓ |

Occupancy follows: 1P1C 4s 1→2 CTAs/SM (smem-bound), 1P1C 3s 2→3,
1P2C/1P3C register-bound at 1 CTA/SM, FA register-bound at 1 CTA/SM —
all as in the paper's table.

Two occupancy *limits* are reported separately, because they differ where
registers bind: for SALA 1P2C/1P3C the shared-memory capacity allows
2 CTAs/SM while the register limit still allows only 1. The achieved
occupancy is the binding minimum (1, matching ncu's reported value); the
paper's Table 3 lists both limits with the binding one in bold, so a
column reading `smem_occ=2` next to `reg_occ=1` means SMEM stopped being
the constraint — not that 2 CTAs are resident.

The Act. column was re-measured during artifact evaluation and the
paper's Table 3 carries the measured values (marked orange in the
review PDF). The script prints them for all rows: GEMM rows at the
kernels' default 2048³ workload (the generated harness runs its
baked-in size; the `--m/--n/--k` arguments are ignored), FA rows at
the paper's pinned config (B=2 H=16, SEQ=16384 — the 6th of the
harness's 6 configs). Expected: 1P1C 4s 7.6→14.8% | 1P1C 3s
14.2→20.4% (observed 20.2–21.4% across runs — the 3-CTA/SM cell
is the most variance-prone) | 1P2C 18.0→18.0% | 1P3C
24.0→24.0% | FA tuned 18.4→18.4% | FA 3s 18.3% (SALA). The 1P1C
doubling (the occupancy claim) and the FA flatness reproduce
exactly; the absolute levels differ from the submission's (the
paper's 1P1C 12.5% is the theoretical 8-warps-of-64 ceiling, not
the achieved value).

---

## 4. Figure 3 (fig:throughput) — End-to-End Ratios (~30 min, needs GPU)

One command — compiles and runs all five configs in both modes at
4096³ (10 warmup + 500 timed iterations, the paper's methodology):

```bash
GPU=0 bash reproduce/figure3_throughput.sh
```

The ratio structure reproduces exactly:

| Config | Kernel | Base TFLOPS | SALA TFLOPS | Ratio | Paper |
|--------|--------|-------------|-------------|-------|-------|
| 4s 64×128 (f16) | `matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co` | 226.9 | 313.1 | **1.38×** | 1.34× |
| 4s 64×128 (f8) | `matmul_e4m3_dyn_sm90_warpspec_1p1c.co` | 370.4 | 497.4 | **1.34×** | 1.40× |
| 3s 64×128 | `matmul_f16_dyn_sm90_warpspec_1p1c_64x128_3s.co` | 301.4 | 314.7 | 1.04× | 1.00× |
| 1P2C GEMM (3s) | `matmul_f16_dyn_sm90_warpspec_1p2c.co` | 248.5 | 251.9 | 1.01× | 1.00× |
| 1P3C GEMM (2s) | `matmul_f16_dyn_sm90_warpspec_1p3c.co` | 275.2 | 273.8 | 0.99× | 1.00× |

Only the two 1P1C 4s configs (crossing the 1→2 CTA/SM boundary) gain;
3s/1P2C/1P3C are flat. Absolute TFLOPS differ from the paper's bars
(machine state); the ratios are the figure's claim and they match.

Per-kernel runs (M=N=K=4096 — sed the `MATMUL_DEFAULT_*` to 4096
before compiling; the kernels are in `benchmarks/matmul/`):

```bash
croqtile/build/choreo -t cute -arch=sm_90a benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co -o /tmp/mm   # + --no-sala for the baseline
CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500 /tmp/mm
```

FA e2e (the paper's ~360 claim): reproduce with the tuned FA kernel
at the AE config (B=2 H=16, SEQ=16384):

```bash
croqtile/build/choreo -t cute -arch=sm_90a benchmarks/flash_atten/fa_tuned_1p2c.co -o /tmp/fa_tuned            # SALA
croqtile/build/choreo -t cute -arch=sm_90a --no-sala benchmarks/flash_atten/fa_tuned_1p2c.co -o /tmp/fa_tuned_nosala  # baseline
CHOREO_TIMING_WARMUP=2 CHOREO_TIMING_REPEAT=3 /tmp/fa_tuned            # SALA
CHOREO_TIMING_WARMUP=2 CHOREO_TIMING_REPEAT=3 /tmp/fa_tuned_nosala     # baseline
```

(Without `-gs`, choreo emits the final binary directly; the harness
verifies by default, so both runs end with `Test Passed`. Each
invocation measures one mode — the harness runs a single sweep per
binary.)

Expected: ~360 TFLOPS in BOTH modes at SEQ=16384 (measured 360.6
SALA / 358.5 no-SALA — flat within ~2%: the run-to-run spread
between the two modes was 0.6-2.0% across the verified runs),
matching the paper's ~360
at its pinned config (B=2 H=16). The flatness — the paper's claim that
the 197→164 KB reduction does not affect FA throughput — holds at
B=1 H=16 (~305 TFLOPS) and B=2 H=16 alike. See §6.2 for the
config-dependence and §6.6 for why the FA row uses short repeats.

---

## 5. Table 4 (tab:ablation) — SALA Mechanisms

The paper's Table 4 covers two experiment sets: the **1P3C GEMM smem**
ablation (kernel compiles, §5.1) and the **FA K/V safety column**
(§5.2). Two 1P3C kernels are committed: `..._1p3c.co` carries
`sync.wg 1, 2, 3;` at the epilogue; `..._1p3c_nobarrier.co` is the
same kernel without it — SALA never inserts barriers, it reads the
kernel's explicit `sync.wg`.

### 5.1 1P3C GEMM smem ablation (kernel compiles)

| Configuration | 1P3C smem | Reproduce |
|---|---|---|
| No SALA (baseline) | 128 KB | `croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p3c_nobarrier.co` → 131072 B |
| +Barrier only | 128 KB | `croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` → 131072 B (the `sync.wg` is present; under standard liveness the barrier alone gives no savings — the row's point) |
| +SALA (phase-disjoint) | 80 KB | `croqtile/build/choreo -gs -t cute -arch=sm_90a benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` → 81920 B (+SALA implicitly includes the barrier: SALA requires and verifies `sync.wg` for multi-consumer kernels — the paper's SALA kernels carry it) |
| +HB w/o acyclic | 80 KB | same command as +SALA → 81920 B (the allocation is identical; the constraint's effect is on the FA K/V safety, §5.2) |
| +HB w/ acyclic | 80 KB | same command as +SALA → 81920 B |

For completeness: the no-barrier kernel compiled with SALA stays at
131072 B — SALA conservatively refuses the overlap without the
barrier, consistent with the paper's claim that SALA never inserts
barriers.

### 5.2 FA K/V safety column (vendored analyzer)

The acyclic constraint's effect on whether FA K_s/V_s may overlap,
demonstrated by the vendored analyzer — a faithful re-implementation
of the analysis's acyclic-constraint decision (the paper describes
this behavior in terms of the analysis itself). Rows 1–2 show the
baseline's allocation-level "no overlap" on the tuned FA kernel;
rows 3–5 use the analyzer on the FA K/V pattern (the same K/V
pipeline structure as Figure 2's item-5 model, 40→32 KB) — the
safety question is structural and independent of the tile size.

| Configuration | FA K/V | Sound? | Reproduce |
|---|---|---|---|
| No SALA (baseline) | no overlap | ✓ | `croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala benchmarks/flash_atten/fa_tuned_1p2c_nobarrier.co` → 196608 B (all four buffers allocated separately — no overlap) |
| +Barrier only | no overlap | ✓ | `croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala benchmarks/flash_atten/fa_tuned_1p2c.co` → 196608 B (the `sync.wg` is present; standard liveness still overlaps nothing) |
| +SALA (phase-disjoint) | no overlap | ✓ | `PYTHONPATH=. python3 -m hb_analyzer pattern hb_analyzer/examples/choreo_fa_kv.json` → 2 safe pairs; (K_s, V_s) non-overlappable |
| +HB w/o acyclic | **overlap** | ✗ (race) | `HBA_NO_ACYCLIC=1 PYTHONPATH=. python3 -m hb_analyzer pattern hb_analyzer/examples/choreo_fa_kv.json` → 3 overlappable pairs, incl. (K_s, V_s) |
| +HB w/ acyclic | no overlap | ✓ | analyzer default → 2 safe pairs; (K_s, V_s) non-overlappable |


---

## 6. Notes for the Reviewer

### 6.1 Register counts vs CUDA minor

On CUDA 13.0 the Table-3 registers are the paper's values. Under a
newer CUDA (e.g., 13.3) the 1P2C/1P3C cells may read 93/96 — `ptxas`
re-allocates registers slightly differently; 1P1C's 72 is stable.
Shared-memory values, occupancy, and correctness are version-stable
across CUDA 13.x. The FA row's 168 regs (paper's value, updated at
camera-ready from the earlier 158) is stable across all available
toolchains.

### 6.2 FA throughput is config-dependent

The tuned FA's TFLOPS varies strongly with the workload's B×H: B=1
H=16 gives ~305 TFLOPS, while the paper's config (B=2 H=16, pinned in
both throughput passages) gives ~360 TFLOPS at SEQ=16384 — matching
the paper's ~360 (short-repeat runs; see §4). The claim to verify is
the *structure*: SALA ≈ no-SALA flat (within ~2%), confirming that
the 197→164 KB reduction does not affect throughput. Compare within
a run, not across machines or configs.

### 6.3 What the scripts do not cover

The Act. column of Table 3 is re-measured by the scripts (table3
GEMM rows at the kernels' default 2048³ workload; FA rows at B=2
H=16 SEQ=16384) and the paper's Table 3 carries the measured values
(§3). The only Table-3 cells not re-measured are the CUTLASS and
Tawa rows (register-bound; SMEM/occupancy are unchanged).

Correctness coverage differs per family, and the scripts say so
explicitly rather than printing a blanket verdict:

- **XComp rows (Figures 2–3, Table 3 GEMM/FA, Table 4 conv)**: full
  CPU-reference checks, both modes (§1.4).
- **CUTLASS rows**: the baseline is verified numerically at 2048³;
  the struct→union build is verified at 1024² (single work tile per
  CTA). At 2048³ the union run prints `NOT CHECKED` — the shared
  epilogue/mainloop storage needs cross-tile producer gating under
  persistent scheduling, which the manual workaround does not
  implement (§2.3). The SMEM numbers are unaffected by this: they are
  a property of the kernel, not of the work assignment.
- **Tawa prototype scope**: the vendored pass is a prototype of the
  interference refinement (`~50 LOC`), not the production analysis; its
  inserted cross-tile barrier covers the GEMM loop shapes at
  `num_warps=4` (other warp counts are rejected at compile time),
  and the FMHA row's synchronization comes from the kernel's mbarrier
  protocol (`--membar 1`). See §2.4.
- **Tawa rows**: SMEM measurements via `ncu`. The **FMHA** outputs are
  additionally checked against a fp32 torch reference in both stages
  and both modes (2 % + 2 % tolerance; the 3-stage baseline cannot
  launch, so only its SALA side is checked) — see §2.4, including the
  `--membar` requirement without which the 3-stage kernel races. The
  **GEMM** rows are checked too: fp32 torch matmul reference, rel. err
  < 5 % (`tawa_sala_config_test.py` prints it and exits nonzero on
  failure).

### 6.4 H100 vs H800: which numbers move

The paper's measurements are from an **H800** (PCIe); all of our own
measurements are H800-only. Two Table-3 / Figure-3 quantities are
machine-sensitive, and the artifact documents them so a reviewer on an
H100 can interpret their own numbers:

- **Act. (`sm__warps_active`) for 1P1C-3s**: 14.2→20.4 % on our H800
  (PCIe). Reported SALA-side values span ≈17.2–20.4 % across parts —
  e.g. 20.4 % on an H100 PCIe (matching ours) and ≈17.2 % on an H800
  NVL — so the absolute percentage tracks the part's power/clock
  budget rather than a clean H100-vs-H800 split. The direction and the
  occupancy step (2→3 CTAs/SM) are what the claim rests on, and they
  reproduce everywhere.
- **Figure-3 ratio for 1P1C-3s**: flat-to-slightly-regressed,
  occupancy/machine-dependent — treat this row as flat, not as a gain.
  Our H800 runs give 0.99–1.04× (the paper's measurement: 0.99×), and
  every H100 measurement reproducible from the reports matches that
  picture: 0.95–0.96×, 0.9514×, 0.957× (132-SM SXM), 1.011× (114-SM
  PCIe). One early report of a ~1.10× H100 gain has not been reproduced
  by us or by later evaluation, and we cannot reconstruct its
  configuration, so we no longer cite it. The occupancy step (2→3
  CTAs/SM: 91.1→74.8 KB against 228 KB/SM, registers permitting 3)
  reproduces on every part. The 1P1C-4s rows (1.34–1.40×) are the
  headline gains and reproduce on both.

### 6.5 Measurement methodology (trials, order, variability)

- **GEMM timed runs (Figure 3)**: 10 warmup iterations + 500 timed
  iterations per kernel, `cudaEvent` timing, baseline and SALA built
  from the same source with only `--no-sala` differing; each config's
  two binaries are run back-to-back in the same process order, and the
  script prints both absolute TFLOPS and the ratio so the ratio can be
  read within a run (absolute values drift with clock/power state).
- **FA e2e (Table 3 / §6.6)**: warmup=2, repeat=3 (see §6.6 for why
  the paper's 10/500 is not used there).
- **ncu rows**: one launch per config, `launch__shared_mem_per_block_dynamic`
  — a static property of the compiled kernel, so it does not vary
  between runs; re-running changes nothing.
- **Run-to-run variability** is therefore confined to the timed rows
  (§4): expect the 1P1C-4s ratios to wander by ~±0.03× and the flat
  rows by ~±0.02×; the SMEM/register/occupancy numbers are exact.

### 6.6 Why the FA row uses short repeats (H800 power budget)

The paper's 10-warmup/500-iteration methodology is used for the GEMM
timed runs (§4); the FA e2e row deliberately uses short repeats
(warmup=2, repeat=3). At SEQ=16384 the tuned FA kernel saturates the
H800 PCIe's 350 W power cap (measured 348 W), which drops the SM
clock from 1755 to 1395 MHz — a sustained loop (e.g.
warmup=10/repeat=500) then measures ~313 TFLOPS instead of the
paper's ~360, and the power-cap clock transitions can occasionally
race the TMA pipeline and hang the kernel (short loops never trigger
it). The paper's number reproduces at short repeats, and the
SALA ≈ no-SALA flatness holds at any repeat — check the flatness,
not the absolute value (compare within a run, not across machines).

---

## 7. Layout

- `croqtile/` — the compiler as a **submodule**, pinned to the
  paper-lineage commit `a9cd1ba` (branch `sala-artifacts` of
  `LancerLab/croqtile`): `lib/`, `runtime/`, `tools/`, `cmake/`,
  `Makefile`, `CMakeLists.txt` — the paper-lineage DSL with
  `mma.commit`/`mma.wait`, `frag` ops, `sync.wg`, and the SALA
  integration, plus the small codegen/liveness fixes the FA/GEMM rows
  require.
- `benchmarks/` — the kernels: `matmul/` (1P1C 64x128 4s/3s,
  e4m3, 1P2C, 1P3C), `flash_atten/` (FA tuned 1P2C, FA 1P1C 2s; the
  3-stage FA is generated by the script via `sed`), `conv/` (the
  im2col convolution kernel), `cutlass/`
  (union-patch + test for the CUTLASS rows), `tawa/` (vendored
  triton-aref for the Tawa rows).
- `reproduce/` — the one-command reproduction scripts.
- `hb_analyzer/` — the HB-graph pattern analyzer (model rows
  and the ablation).
- `setup_cutlass.sh` — the one-time CUTLASS v4.5.0 headers + patch
  setup for the CUTLASS rows.
- `Dockerfile` — the AE container (builds the compiler from
  `croqtile/` inside the image); `README.md` — this guide.
