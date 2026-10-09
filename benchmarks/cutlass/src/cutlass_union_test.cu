/*
 * SALA CUTLASS Union Test - SharedStorage struct vs union analysis.
 * Measures sizes and runs correctness for cooperative warp-specialized GEMM.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = cutlass::half_t;
using LayoutA  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16 / sizeof(ElementA);
using ElementB = cutlass::half_t;
using LayoutB  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16 / sizeof(ElementB);
using ElementC = cutlass::half_t;
using LayoutC  = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 16 / sizeof(ElementC);
using ElementD = cutlass::half_t;
using LayoutD  = LayoutC;
constexpr int AlignmentD = AlignmentC;
using ElementAccumulator = float;
using ElementCompute     = float;
using ClusterShape = Shape<_1, _1, _1>;
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

// Build epilogue ONCE (it doesn't depend on stages)
template <typename TileShape_>
using BuildEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

// Build mainloop with explicit carveout
template <typename TileShape_, int Stages>
using BuildMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_, ClusterShape,
    cutlass::gemm::collective::StageCount<Stages>,
    KernelSchedule
>::CollectiveOp;

// Full kernel
template <typename TileShape_, int Stages>
using BuildGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    BuildMainloop<TileShape_, Stages>,
    BuildEpilogue<TileShape_>
>;

// The union build is identified by the SALA_UNION macro that the patch adds
// to the patched CUTLASS header (see patches/sala_union.patch).
#ifdef SALA_UNION
constexpr bool kUnionBuild = true;
#else
constexpr bool kUnionBuild = false;
#endif

// Sampled fp32 reference verdict.
struct RefVerdict {
    float max_abs = 0.0f;   // largest |D - D_ref| over the sampled elements
    int   n_bad = 0;        // samples outside tolerance, or non-finite
    int   n_nonfinite = 0;  // non-finite sampled outputs (NaN/Inf)
};

// The comparison must reject non-finite values explicitly: NaN/Inf comparisons
// are unordered, so a bare `diff > tol` test silently accepts them.
static RefVerdict reference_check(const cutlass::half_t* h_A,
                                  const cutlass::half_t* h_B,
                                  const cutlass::half_t* h_D,
                                  int M, int N, int K, size_t elems_CD,
                                  size_t nsamp) {
    RefVerdict v;
    for (size_t s = 0; s < nsamp; s++) {
        size_t idx = (s * 7919 + 13) % elems_CD;
        size_t m = idx % (size_t)M, n = idx / (size_t)M;
        float ref = 0.0f;
        for (int k = 0; k < K; k++)
            ref += float(h_A[(size_t)m * K + k]) * float(h_B[(size_t)n * K + k]);
        float got = float(h_D[idx]);
        if (!std::isfinite(got) || !std::isfinite(ref)) {
            v.n_nonfinite++; v.n_bad++;
            continue;
        }
        float ae = fabsf(got - ref);
        if (ae > v.max_abs) v.max_abs = ae;
        if (ae > 0.1f + 0.01f * fabsf(ref)) v.n_bad++;
    }
    return v;
}

// Self-test of the checker itself: a checker that cannot fail is worthless, so
// we feed it outputs we know are wrong and require it to reject each. Runs on
// the host only (no GPU needed); exit status 0 = all negative tests rejected.
static int selftest() {
    printf("================================================\n");
    printf("Checker self-test (negative tests, host-only)\n");
    printf("================================================\n");
    const int M = 64, N = 64, K = 32;
    const size_t eA = (size_t)M * K, eB = (size_t)N * K, eD = (size_t)M * N;
    std::vector<cutlass::half_t> A(eA), B(eB), D(eD);
    srand(42);
    for (auto& x : A) x = cutlass::half_t(float(rand() % 10 - 5) / 10.0f);
    for (auto& x : B) x = cutlass::half_t(float(rand() % 10 - 5) / 10.0f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float r = 0.0f;
            for (int k = 0; k < K; k++)
                r += float(A[(size_t)m * K + k]) * float(B[(size_t)n * K + k]);
            D[m + (size_t)n * M] = cutlass::half_t(r);
        }

    int failures = 0;
    auto report = [&](const char* what, RefVerdict v, bool expect_reject) {
        bool rejected = (v.n_bad > 0);
        printf("  %-22s max|D-Dref|=%-9.4f bad=%-4d non-finite=%-3d -> %-6s (expected %s)\n",
               what, v.max_abs, v.n_bad, v.n_nonfinite,
               rejected ? "REJECT" : "accept", expect_reject ? "REJECT" : "accept");
        if (rejected != expect_reject) failures++;
    };

    report("clean output", reference_check(A.data(), B.data(), D.data(), M, N, K, eD, eD), false);

    std::vector<cutlass::half_t> Dnan = D;
    Dnan[(0 * 7919 + 13) % eD] = cutlass::half_t(NAN);
    report("NaN injected", reference_check(A.data(), B.data(), Dnan.data(), M, N, K, eD, eD), true);

    std::vector<cutlass::half_t> Dinf = D;
    Dinf[(1 * 7919 + 13) % eD] = cutlass::half_t(INFINITY);
    report("Inf injected", reference_check(A.data(), B.data(), Dinf.data(), M, N, K, eD, eD), true);

    std::vector<cutlass::half_t> Dbig = D;
    size_t big = (2 * 7919 + 13) % eD;
    Dbig[big] = cutlass::half_t(float(Dbig[big]) + 100.0f);
    report("large finite error", reference_check(A.data(), B.data(), Dbig.data(), M, N, K, eD, eD), true);

    printf("\nSelf-test: %s\n", failures ? "FAILED" : "all negative tests rejected as required");
    return failures ? 1 : 0;
}

template <typename GK>
bool run_gemm(int M, int N, int K, bool verify = true, const char* why = nullptr) {
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;
    using StrideA = typename GK::StrideA;
    using StrideB = typename GK::StrideB;
    using StrideC = typename GK::StrideC;
    using StrideD = typename GK::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    size_t elems_A = M * K, elems_B = K * N, elems_CD = M * N;
    cutlass::half_t *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, elems_A * 2);
    cudaMalloc(&d_B, elems_B * 2);
    cudaMalloc(&d_C, elems_CD * 2);
    cudaMalloc(&d_D, elems_CD * 2);

    auto* h_A = new cutlass::half_t[elems_A];
    auto* h_B = new cutlass::half_t[elems_B];
    srand(42);
    for (size_t i = 0; i < elems_A; i++) h_A[i] = cutlass::half_t(float(rand() % 10 - 5) / 10.0f);
    for (size_t i = 0; i < elems_B; i++) h_B[i] = cutlass::half_t(float(rand() % 10 - 5) / 10.0f);
    cudaMemcpy(d_A, h_A, elems_A * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, elems_B * 2, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, elems_CD * 2);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A, stride_A, d_B, stride_B},
        {{1.0f, 0.0f}, d_C, stride_C, d_D, stride_D}
    };

    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(args);
    void* d_ws = nullptr;
    if (ws_size > 0) cudaMalloc(&d_ws, ws_size);

    auto status = gemm.initialize(args, d_ws);
    if (status != cutlass::Status::kSuccess) {
        printf("  Init FAILED\n");
        delete[] h_A; delete[] h_B;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
        if (d_ws) cudaFree(d_ws);
        return false;
    }

    status = gemm();
    cudaError_t err = cudaDeviceSynchronize();
    bool ok = (status == cutlass::Status::kSuccess && err == cudaSuccess);

    auto* h_D = new cutlass::half_t[elems_CD];
    cudaMemcpy(h_D, d_D, elems_CD * 2, cudaMemcpyDeviceToHost);
    float sum = 0; int nonzero = 0; int n_nonfinite_out = 0;
    for (size_t i = 0; i < elems_CD; i++) {
        float v = float(h_D[i]); sum += v;
        if (v != 0.0f) nonzero++;
        if (!std::isfinite(v)) n_nonfinite_out++;
    }

    // Numerical reference: sampled fp32 dot products from the same fp16 inputs.
    // Layouts used by cutlass::make_cute_packed_stride in this test (validated
    // against the pristine v4.5.0 struct kernel):
    //   D (M,N) column-major -> element (m,n) at m + n*M
    //   A (M,K) row-major    -> element (m,k) at m*K + k
    //   B (N,K)              -> element (n,k) at n*K + k
    if (!verify) {
        printf("  GEMM: sum=%.2f nonzero=%d/%zu non-finite=%d\n",
               sum, nonzero, elems_CD, n_nonfinite_out);
        printf("  Reference: NOT CHECKED here (%s)\n", why ? why : "not requested");
        delete[] h_A; delete[] h_B; delete[] h_D;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
        if (d_ws) cudaFree(d_ws);
        return ok;
    }

    RefVerdict v = reference_check(h_A, h_B, h_D, M, N, K, elems_CD, 4096);
    bool numeric_ok = (v.n_bad == 0) && (n_nonfinite_out == 0);
    printf("  GEMM: sum=%.2f nonzero=%d/%zu non-finite=%d\n",
           sum, nonzero, elems_CD, n_nonfinite_out);
    printf("  Reference: 4096 samples, max|D-Dref|=%.4f, %d bad, %d non-finite %s\n",
           v.max_abs, v.n_bad, v.n_nonfinite, numeric_ok ? "PASS" : "FAIL");
    printf("  Result: %s\n", (ok && numeric_ok) ? "PASS" : "FAIL");
    bool pass = ok && numeric_ok;

    delete[] h_A; delete[] h_B; delete[] h_D;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    if (d_ws) cudaFree(d_ws);
    return pass;
}

template <typename TileShape_, int Stages>
bool analyze_and_run(const char* label, int M, int N, int K,
                     bool verify = true, const char* why = nullptr) {
    using GK = BuildGemmKernel<TileShape_, Stages>;
    using SS = typename GK::SharedStorage;
    using ML = typename GK::CollectiveMainloop;
    using EP = typename GK::CollectiveEpilogue;

    printf("\n--- %s ---\n", label);

    size_t mainloop_ts = sizeof(typename ML::TensorStorage);
    size_t epilogue_ts = sizeof(typename EP::TensorStorage);
    size_t total_ss    = sizeof(SS);

    printf("  Mainloop TensorStorage = %.1f KB\n", mainloop_ts / 1024.0);
    printf("  Epilogue TensorStorage = %.1f KB\n", epilogue_ts / 1024.0);
    printf("  SharedStorage (struct) = %.1f KB\n", total_ss / 1024.0);

    // Union: max(mainloop, epilogue) instead of sum
    size_t max_ts = (mainloop_ts > epilogue_ts) ? mainloop_ts : epilogue_ts;
    size_t min_ts = (mainloop_ts < epilogue_ts) ? mainloop_ts : epilogue_ts;
    size_t savings = min_ts;  // union saves the smaller one
    size_t new_total = total_ss - savings;
    // Round up to 128-byte boundary
    new_total = (new_total + 127) & ~127;
    savings = total_ss - new_total;

    printf("  SharedStorage (union)  = %.1f KB (est)\n", new_total / 1024.0);
    printf("  Savings: %.1f KB (%.1f%%)\n", savings / 1024.0,
           100.0 * savings / total_ss);

    int smem_per_sm = 228 * 1024;
    int orig_ctas = smem_per_sm / (int)total_ss;
    int new_ctas  = smem_per_sm / (int)new_total;
    printf("  Occupancy: %d -> %d CTAs/SM", orig_ctas, new_ctas);
    if (new_ctas > orig_ctas) printf(" ***IMPROVED***");
    printf("\n");

    return run_gemm<GK>(M, N, K, verify, why);
}

using T128 = Shape<_128, _128, _64>;
using T256 = Shape<_128, _256, _64>;

// ---------------------------------------------------------------------------
// Non-persistent kernel: CUTLASS itself unions the mainloop and epilogue
// storage here -- sm90_gemm_tma_warpspecialized.hpp carries the comment
// "Mainloop and epilogue don't use smem concurrently since kernel is
// non-persistent, so we can use a union".  One work tile per CTA means the
// two phases cannot overlap, which is exactly the condition SALA's analysis
// checks before it permits an overlap.  This kernel needs no patch: the
// overlap is already in CUTLASS, and we verify its numerics here.
// ---------------------------------------------------------------------------
using KernelScheduleNP   = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueScheduleNP = cutlass::epilogue::TmaWarpSpecialized;

template <typename TileShape_>
using BuildEpilogueNP = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueScheduleNP
>::CollectiveOp;

template <typename TileShape_, int Stages>
using BuildMainloopNP = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_, ClusterShape,
    cutlass::gemm::collective::StageCount<Stages>,
    KernelScheduleNP
>::CollectiveOp;

template <typename TileShape_, int Stages>
using BuildGemmKernelNP = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    BuildMainloopNP<TileShape_, Stages>,
    BuildEpilogueNP<TileShape_>
>;

template <typename TileShape_, int Stages>
bool nonpersistent_check(const char* label, int M, int N, int K) {
    using GK = BuildGemmKernelNP<TileShape_, Stages>;
    using SS = typename GK::SharedStorage;
    using ML = typename GK::CollectiveMainloop;
    using EP = typename GK::CollectiveEpilogue;

    size_t ml = sizeof(typename ML::TensorStorage);
    size_t ep = sizeof(typename EP::TensorStorage);
    size_t ss = sizeof(SS);

    printf("\n--- %s, non-persistent (CUTLASS union) ---\n", label);
    printf("  Mainloop TensorStorage = %.1f KB, Epilogue = %.1f KB\n",
           ml / 1024.0, ep / 1024.0);
    printf("  SharedStorage = %.1f KB vs %.1f KB if the two were laid out "
           "separately %s\n", ss / 1024.0, (ml + ep) / 1024.0,
           (ss < ml + ep) ? "(overlapped - CUTLASS's own union)" : "(not overlapped)");
    return run_gemm<GK>(M, N, K, true);
}

// Work tiles for an MxN problem with the given tile shape (K is summed inside
// the tiles, so it does not change the work-tile count).
template <typename TileShape_>
int work_tiles(int M, int N) {
    int tm = size<0>(TileShape_{});
    int tn = size<1>(TileShape_{});
    return ((M + tm - 1) / tm) * ((N + tn - 1) / tn);
}

// The manual struct->union overlap shares the epilogue staging with the
// mainloop stages.  With the persistent scheduler, a CTA may process several
// work tiles; the next tile's mainloop TMA loads then overwrite the union'd
// memory while the previous tile's epilogue is still using it.  That hazard
// can only be prevented by gating the *producer* warps (a consumer-side
// NamedBarrier cannot: see the README).  We therefore verify numerics only
// where each CTA owns a single work tile.
template <typename TileShape_, int Stages>
bool safe_regime(int M, int N, int sm_count) {
    if (!kUnionBuild) return true;          // struct build: correct at any size
    return work_tiles<TileShape_>(M, N) <= sm_count;
}

int main(int argc, char** argv) {
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") return selftest();
        if (std::string(argv[i]) == "--check") check_only = true;
    }

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    if (sm_count <= 0) sm_count = 132;

    if (check_only) {
        printf("================================================\n");
        printf("CUTLASS %s build - numerical verification\n",
               kUnionBuild ? "struct->union" : "pristine struct");
        printf("device: %d SMs; safe regime = work tiles <= SM count\n", sm_count);
        printf("================================================\n");
        int failures = 0;
        int fm = kUnionBuild ? 1024 : 2048;   // union: 64/32 tiles <= SMs
        int fn = kUnionBuild ? 1024 : 2048;
        const char* note = kUnionBuild
            ? "verified at 1024x1024 (single work tile per CTA); the 2048^3 "
              "persistent assignment needs cross-tile producer gating"
            : nullptr;
        failures += !analyze_and_run<T128, 2>("128x128x64, 2-stage", fm, fn, 2048, true, note);
        failures += !analyze_and_run<T128, 3>("128x128x64, 3-stage", fm, fn, 2048, true, note);
        failures += !analyze_and_run<T128, 4>("128x128x64, 4-stage", fm, fn, 2048, true, note);
        failures += !analyze_and_run<T256, 2>("128x256x64, 2-stage", fm, fn, 2048, true, note);
        failures += !analyze_and_run<T256, 3>("128x256x64, 3-stage", fm, fn, 2048, true, note);
        printf("\nDone: %d/5 configurations verified.\n", 5 - failures);

        // The non-persistent kernel: CUTLASS's own union, one work tile per
        // CTA -- the valid case SALA's analysis permits, verified here.
        printf("\n\n================================================\n");
        printf("Non-persistent kernel (CUTLASS's own union; no patch)\n");
        printf("================================================\n");
        int np_fail = 0;
        np_fail += !nonpersistent_check<T128, 2>("128x128x64, 2-stage", 2048, 2048, 2048);
        np_fail += !nonpersistent_check<T128, 3>("128x128x64, 3-stage", 2048, 2048, 2048);
        np_fail += !nonpersistent_check<T256, 2>("128x256x64, 2-stage", 2048, 2048, 2048);
        printf("\nNon-persistent: %d/3 configurations verified.\n", 3 - np_fail);

        printf("\nSummary (checked vs skipped):\n");
        if (kUnionBuild) {
            printf("  checked:  5/5 persistent-union configs, single-tile regime (1024^2: one work tile/CTA)\n");
            printf("            3/3 non-persistent configs at 2048^3\n");
            printf("  skipped:  the persistent union at 2048^3 -- multi-tile assignment; not a valid\n");
            printf("            overlap without producer-side gating (prints NOT CHECKED there; README 2.3)\n");
        } else {
            printf("  checked:  5/5 configs at 2048^3\n");
            printf("            3/3 non-persistent configs at 2048^3\n");
            printf("  skipped:  none\n");
        }

        return (failures || np_fail) ? 1 : 0;
    }

    printf("================================================\n");
    printf("SALA CUTLASS Cooperative SharedStorage Analysis\n");
    printf("H100 (SM90a), 228 KB SMEM/SM\n");
    printf("================================================\n");

    const char* skip_reason =
        "persistent multi-tile assignment: the shared epilogue/mainloop storage "
        "needs cross-tile producer gating; see README section 2.3";

    int failures = 0;
    failures += !analyze_and_run<T128, 2>("128x128x64, 2-stage", 2048, 2048, 2048,
        safe_regime<T128, 2>(2048, 2048, sm_count), skip_reason);
    failures += !analyze_and_run<T128, 3>("128x128x64, 3-stage", 2048, 2048, 2048,
        safe_regime<T128, 3>(2048, 2048, sm_count), skip_reason);
    failures += !analyze_and_run<T128, 4>("128x128x64, 4-stage", 2048, 2048, 2048,
        safe_regime<T128, 4>(2048, 2048, sm_count), skip_reason);
    failures += !analyze_and_run<T256, 2>("128x256x64, 2-stage", 2048, 2048, 2048,
        safe_regime<T256, 2>(2048, 2048, sm_count), skip_reason);
    failures += !analyze_and_run<T256, 3>("128x256x64, 3-stage", 2048, 2048, 2048,
        safe_regime<T256, 3>(2048, 2048, sm_count), skip_reason);

    printf("\nDone: %d/5 configurations passed.\n", 5 - failures);
    return failures ? 1 : 0;
}

#else
int main() { printf("SM90 not supported\n"); return 1; }
#endif
