#include "triton/Analysis/Allocation.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Alias.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "allocation-shared-memory"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

//===----------------------------------------------------------------------===//
// Shared Memory Allocation Analysis
//===----------------------------------------------------------------------===//
namespace triton {

// Bitwidth of pointers
constexpr int kPtrBitWidth = 64;
// Max shmem LDS/STS instruction in bits
constexpr int kMaxShmemVecBitLength = 128;

static SmallVector<unsigned> getRepShapeForCvt(RankedTensorType srcTy,
                                               RankedTensorType dstTy) {
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  if (!cvtNeedsSharedMemory(srcTy, dstTy)) {
    return {};
  }

  if (shouldUseDistSmem(srcLayout, dstLayout)) {
    // TODO: padding to avoid bank conflicts
    return convertType<unsigned, int64_t>(gpu::getShapePerCTA(srcTy));
  }

  assert(srcLayout && dstLayout && "Unexpected layout in getRepShapeForCvt()");

  auto srcShapePerCTA = gpu::getShapePerCTA(srcTy);
  auto dstShapePerCTA = gpu::getShapePerCTA(dstTy);
  auto srcShapePerCTATile = gpu::getShapePerCTATile(srcTy);
  auto dstShapePerCTATile = gpu::getShapePerCTATile(dstTy);

  assert(srcTy.getRank() == dstTy.getRank() &&
         "src and dst must have the same rank");

  unsigned rank = dstTy.getRank();
  SmallVector<unsigned> repShape(rank);
  for (unsigned d = 0; d < rank; ++d) {
    repShape[d] =
        std::max(std::min<unsigned>(srcShapePerCTA[d], srcShapePerCTATile[d]),
                 std::min<unsigned>(dstShapePerCTA[d], dstShapePerCTATile[d]));
  }
  return repShape;
}

// Both `atomic_cas` and `atomic_rmw need a single scratch element if returning
// a scalar value because Triton's block-based programming model ensures that
// all threads in each block see the same return value, even those threads that
// do not participate in the atomic operation
static SmallVector<unsigned> getRepShapeForAtomic(Value result) {
  SmallVector<unsigned> smemShape;
  if (atomicNeedsSharedMemory(result)) {
    smemShape.push_back(1);
  }
  return smemShape;
}

std::pair<unsigned, unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy) {
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  auto srcLinAttr = gpu::toLinearEncoding(srcLayout, srcTy.getShape());
  auto dstLinAttr = gpu::toLinearEncoding(dstLayout, dstTy.getShape());
  auto inOrd = srcLinAttr.getOrder();
  auto outOrd = dstLinAttr.getOrder();

  unsigned rank = srcTy.getRank();

  unsigned srcContigPerThread = srcLinAttr.getContigPerThread()[inOrd[0]];
  unsigned dstContigPerThread = dstLinAttr.getContigPerThread()[outOrd[0]];
  // TODO: Fix the legacy issue that outOrd[0] == 0 always means
  //       that we cannot do vectorization.
  unsigned innerDim = rank - 1;
  unsigned inVec = outOrd[0] != innerDim  ? 1
                   : inOrd[0] != innerDim ? 1
                                          : srcContigPerThread;
  unsigned outVec = outOrd[0] != innerDim ? 1 : dstContigPerThread;

  if (isa<gpu::NvidiaMmaEncodingAttr>(srcLayout) &&
      isa<gpu::BlockedEncodingAttr>(dstLayout)) {
    // when storing from mma layout and loading in blocked layout vectorizing
    // the load back gives better performance even if there is a
    // transposition.
    outVec = dstContigPerThread;
  }
  return {inVec, outVec};
}

ScratchConfig getScratchConfigForCvt(RankedTensorType srcTy,
                                     RankedTensorType dstTy) {
  // Initialize vector sizes and stride
  auto repShape = getRepShapeForCvt(srcTy, dstTy);
  if (repShape.empty())
    return ScratchConfig({}, {});
  ScratchConfig scratchConfig(repShape, repShape);
  auto rank = repShape.size();
  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();

  assert(cvtNeedsSharedMemory(srcTy, dstTy));
  auto outOrd = gpu::getOrder(dstTy);
  scratchConfig.order = outOrd;

  std::tie(scratchConfig.inVec, scratchConfig.outVec) =
      getScratchCvtInOutVecLengths(srcTy, dstTy);
  // We can't write a longer vector than the shape of shared memory.
  // This shape might be smaller than the tensor shape in case we decided to
  // do the conversion in multiple iterations.
  unsigned contiguousShapeDim = scratchConfig.repShape[scratchConfig.order[0]];
  scratchConfig.inVec = std::min(scratchConfig.inVec, contiguousShapeDim);
  scratchConfig.outVec = std::min(scratchConfig.outVec, contiguousShapeDim);
  // Clamp the vector length to kMaxShmemVecBitLength / element bitwidth as this
  // is the max vectorisation
  auto inBitWidth = isa<PointerType>(srcTy.getElementType())
                        ? kPtrBitWidth
                        : srcTy.getElementTypeBitWidth();
  auto outBitWidth = isa<PointerType>(dstTy.getElementType())
                         ? kPtrBitWidth
                         : dstTy.getElementTypeBitWidth();
  scratchConfig.inVec =
      std::min(scratchConfig.inVec, kMaxShmemVecBitLength / inBitWidth);
  scratchConfig.outVec =
      std::min(scratchConfig.outVec, kMaxShmemVecBitLength / outBitWidth);

  // No padding is required if the tensor is 1-D, or if all dimensions except
  // the first accessed dimension have a size of 1.
  if (rank <= 1 || product(repShape) == repShape[outOrd[0]])
    return scratchConfig;

  auto paddedSize = std::max(scratchConfig.inVec, scratchConfig.outVec);
  scratchConfig.paddedRepShape[outOrd[0]] += paddedSize;
  return scratchConfig;
}

unsigned defaultAllocationAnalysisScratchSizeFn(Operation *op) {
  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    ReduceOpHelper helper(reduceOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto scanOp = dyn_cast<ScanOp>(op)) {
    ScanLoweringHelper helper(scanOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto gatherOp = dyn_cast<GatherOp>(op)) {
    GatherLoweringHelper helper(gatherOp);
    return helper.getScratchSizeInBytes();
  }
  if (auto histogram = dyn_cast<HistogramOp>(op)) {
    auto dstTy = histogram.getType();
    int threadsPerWarp = gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    return std::max<int>(dstTy.getNumElements(), threadsPerWarp) *
           std::max<int>(8, dstTy.getElementTypeBitWidth()) / 8;
  }
  if (auto cvtLayout = dyn_cast<gpu::ConvertLayoutOp>(op)) {
    auto srcTy = cvtLayout.getSrc().getType();
    auto dstTy = cvtLayout.getType();
    auto srcEncoding = srcTy.getEncoding();
    auto dstEncoding = dstTy.getEncoding();
    if (mlir::isa<gpu::SharedEncodingTrait>(srcEncoding) ||
        mlir::isa<gpu::SharedEncodingTrait>(dstEncoding)) {
      // Conversions from/to shared memory do not need scratch memory.
      return 0;
    }
    // ConvertLayoutOp with both input/output non-shared_layout
    // TODO: Besides of implementing ConvertLayoutOp via shared memory, it's
    //       also possible to realize it with other approaches in restricted
    //       conditions, such as warp-shuffle
    auto scratchConfig = getScratchConfigForCvt(srcTy, dstTy);
    auto elems = getNumScratchElements(scratchConfig.paddedRepShape);
    return isa<PointerType>(srcTy.getElementType())
               ? elems * kPtrBitWidth / 8
               : elems * std::max<int>(8, srcTy.getElementTypeBitWidth()) / 8;
  }
  if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
    auto value = op->getOperand(0);
    // only scalar requires scratch memory
    // make it explicit for readability
    if (dyn_cast<RankedTensorType>(value.getType())) {
      return 0;
    }
    auto smemShape = getRepShapeForAtomic(op->getResult(0));
    auto elems = getNumScratchElements(smemShape);
    auto elemTy = cast<PointerType>(value.getType()).getPointeeType();
    assert(!isa<PointerType>(elemTy) && "unexpected pointer type");
    return elems * std::max<int>(8, elemTy.getIntOrFloatBitWidth()) / 8;
  }
  if (isa<ExperimentalTensormapCreateOp>(op)) {
    constexpr int32_t kTMASize = 128;
    return kTMASize;
  }
  return 0;
}

class AllocationAnalysis {
public:
  AllocationAnalysis(Operation *operation,
                     Allocation::FuncAllocMapT *funcAllocMap,
                     Allocation *allocation,
                     AllocationAnalysisScratchSizeFn scratchSizeGetter)
      : operation(operation), funcAllocMap(funcAllocMap),
        allocation(allocation), scratchSizeGetter(scratchSizeGetter) {
    const char *env = std::getenv("SALA_ENABLE");
    salaEnabled = env && std::string(env) == "1";
    const char *dumpEnv = std::getenv("SALA_DUMP_IGRAPH");
    salaDumpIGraph = dumpEnv && std::string(dumpEnv) == "1";
    const char *noBarEnv = std::getenv("SALA_NO_BARRIER");
    salaNoBarrier = noBarEnv && std::string(noBarEnv) == "1";
    run();
  }

private:
  using BufferT = Allocation::BufferT;

  /// Value -> Liveness Range
  /// Use MapVector to ensure determinism.
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
  /// Nodes -> Nodes
  using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;
  /// SALA: skipped interference edges for diagnostics
  using EdgeSetT = DenseSet<std::pair<BufferT *, BufferT *>>;

  bool salaEnabled = false;
  bool salaDumpIGraph = false;
  bool salaNoBarrier = false;
  EdgeSetT salaSkippedEdges;

  void run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
  }

  /// Initializes explicitly defined shared memory values for a given operation.
  void getExplicitValueSize(Operation *op) {
    auto alloc = dyn_cast<gpu::LocalAllocOp>(op);
    if (!alloc || !alloc.isSharedMemoryAlloc())
      return;
    // Bytes could be a different value once we support padding or other
    // allocation policies.
    auto allocType = alloc.getType();
    auto shapePerCTA = gpu::getAllocationShapePerCTA(allocType);
    auto bytes =
        product<int64_t>(shapePerCTA) * allocType.getElementTypeBitWidth() / 8;

    auto alignment = alloc.getAlignmentOrDefault();
    allocation->addBuffer<BufferT::BufferKind::Explicit>(alloc, bytes,
                                                         alignment);
  }

  template <BufferT::BufferKind T>
  void maybeAddScratchBuffer(Operation *op, unsigned bytes,
                             unsigned alignment) {
    if (bytes > 0)
      allocation->addBuffer<T>(op, bytes, alignment);
  }

  template <BufferT::BufferKind T>
  void maybeAddScratchBuffer(Operation *op, unsigned bytes) {
    if (bytes > 0)
      allocation->addBuffer<T>(op, bytes);
  }

  /// Initializes temporary shared memory for a given operation.
  void getScratchValueSize(Operation *op) {
    constexpr size_t scratchAlignment = 128;
    if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      auto callable = callOp.resolveCallable();
      auto funcOp = dyn_cast<FunctionOpInterface>(callable);
      auto *funcAlloc = &(*funcAllocMap)[funcOp];
      auto bytes = funcAlloc->getSharedMemorySize();
      maybeAddScratchBuffer<BufferT::BufferKind::Virtual>(op, bytes,
                                                          scratchAlignment);
      return;
    }
    if (auto ws = dyn_cast<gpu::WarpSpecializeOp>(op)) {
      // `ttg.warp_specialize` needs memory to pass its explicit captures. Pack
      // the captures like a struct.
      auto [captureSize, captureAlign] = ws.getCaptureSizeAlign();
      maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, captureSize,
                                                          captureAlign);
      return;
    }
    if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      unsigned numWarpIndices = 0;
      // Warp specialization communicates states over shared memory to each
      // warp. Add space for an i8 for each warpgroup warp.
      func.walk([&](gpu::WarpSpecializeOp op) {
        numWarpIndices = std::max(numWarpIndices, op.getTotalPartitionWarps());
      });
      maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, numWarpIndices);
      return;
    }
    unsigned bytes = scratchSizeGetter(op);
    maybeAddScratchBuffer<BufferT::BufferKind::Scratch>(op, bytes,
                                                        scratchAlignment);
  }

  void getValueAlias(Value value, SharedMemoryAliasAnalysis &analysis) {
    dataflow::Lattice<AliasInfo> *latticeElement =
        analysis.getLatticeElement(value);
    if (latticeElement) {
      AliasInfo &info = latticeElement->getValue();
      if (!info.getAllocs().empty()) {
        for (auto alloc : info.getAllocs()) {
          allocation->addAlias(value, alloc);
        }
      }
    }
  }

  /// Extract all shared memory values and their sizes
  void getValuesAndSizes() {
    // Get the alloc values
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      getExplicitValueSize(op);
      getScratchValueSize(op);
    });
    // Get the alias values
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    SharedMemoryAliasAnalysis *aliasAnalysis =
        solver->load<SharedMemoryAliasAnalysis>();
    // Run the analysis rooted at every isolated from above operation, including
    // the top-level function but also any nested regions.
    operation->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      if (op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
          failed(solver->initializeAndRun(op))) {
        // TODO: return error instead of bailing out..
        llvm_unreachable("failed to run SharedMemoryAliasAnalysis");
      }
    });
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        getValueAlias(operand, *aliasAnalysis);
      }
      for (auto value : op->getResults()) {
        getValueAlias(value, *aliasAnalysis);
      }
    });
  }

  /// Computes the liveness range of the allocated value.
  /// Each buffer is allocated only once.
  void resolveExplicitBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      bufferRange[buffer] = getLiveness(value);
      LLVM_DEBUG({
        llvm::dbgs() << "-- buffer " << buffer->id << "; value: ";
        value.dump();
      });
    }
  }

  /// Extends the liveness range by unionizing the liveness range of the aliased
  /// values because each allocated buffer could be an alias of others, if block
  /// arguments are involved.
  void resolveAliasBufferLiveness(
      function_ref<Interval<size_t>(Value value)> getLiveness) {
    for (const auto &[value, buffers] : allocation->aliasBuffer) {
      auto range = getLiveness(value);
      for (auto *buffer : buffers) {
        auto minId = range.start();
        auto maxId = range.end();
        if (bufferRange.count(buffer)) {
          // Extend the allocated buffer's range
          minId = std::min(minId, bufferRange[buffer].start());
          maxId = std::max(maxId, bufferRange[buffer].end());
        }
        bufferRange[buffer] = Interval(minId, maxId);
      }
    }
  }

  /// Computes the liveness range of scratched buffers.
  /// Some operations may have a temporary buffer that is not explicitly
  /// allocated, but is used to store intermediate results.
  void resolveScratchBufferLiveness(
      const DenseMap<Operation *, size_t> &operationId) {
    // Analyze liveness of scratch buffers and virtual buffers.
    auto maxInterval = Interval(size_t(), std::numeric_limits<size_t>::max());
    auto processScratchMemory = [&](const auto &container) {
      for (auto [op, buffer] : container) {
        // Buffers owned by the function are assumed live for the whole
        // function. This memory is used for warp specialization codegen.
        // FIXME: Spooky-action-at-a-distance. Find a better way to model this.
        if (op == operation) {
          bufferRange.insert({buffer, maxInterval});
          continue;
        }

        // Any scratch memory's live range is the current operation's live
        // range.
        bufferRange.insert(
            {buffer, Interval(operationId.at(op), operationId.at(op) + 1)});
        LLVM_DEBUG({
          llvm::dbgs() << "-- buffer " << buffer->id << "; value: ";
          op->dump();
        });
      }
    };
    processScratchMemory(allocation->opScratch);
    processScratchMemory(allocation->opVirtual);
  }

  /// Resolves liveness of all values involved under the root operation.
  void resolveLiveness() {
    // Assign an ID to each operation using post-order traversal.
    // To achieve the correct liveness range, the parent operation's ID
    // should be greater than each of its child operation's ID .
    // Example:
    //     ...
    //     %5 = triton.convert_layout %4
    //     %6 = scf.for ... iter_args(%arg0 = %0) -> (i32) {
    //       %2 = triton.convert_layout %5
    //       ...
    //       scf.yield %arg0
    //     }
    // For example, %5 is defined in the parent region and used in
    // the child region, and is not passed as a block argument.
    // %6 should should have an ID greater than its child operations,
    // otherwise %5 liveness range ends before the child operation's liveness
    // range ends.

    // triton::nvidia_gpu::WarpGroup ops do not participate in liveness analysis
    // as these are warp-group regions to facilitate LLVM lowering later
    auto isWarpGroupOp = [](Operation *op) {
      return isa<triton::nvidia_gpu::WarpGroupOp>(op) ||
             isa<triton::nvidia_gpu::WarpGroupReturnOp>(op);
    };

    DenseMap<Operation *, size_t> operationId;
    operation->walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (isWarpGroupOp(op))
        return;
      operationId[op] = operationId.size();
    });

    // Analyze liveness of explicit buffers
    Liveness liveness(operation);
    auto getValueLivenessRange = [&](Value value) {
      auto liveOperations = liveness.resolveLiveness(value);
      auto minId = std::numeric_limits<size_t>::max();
      auto maxId = std::numeric_limits<size_t>::min();
      llvm::for_each(liveOperations, [&](Operation *liveOp) {
        if (isWarpGroupOp(liveOp))
          return;
        if (operationId[liveOp] < minId) {
          minId = operationId[liveOp];
        }
        if ((operationId[liveOp] + 1) > maxId) {
          maxId = operationId[liveOp] + 1;
        }
      });
      return Interval(minId, maxId);
    };

    resolveExplicitBufferLiveness(getValueLivenessRange);
    resolveAliasBufferLiveness(getValueLivenessRange);
    resolveScratchBufferLiveness(operationId);
  }

  void dumpBuffers() const {
    LDBG("Dump bufferRange: id size offset ---------");
    for (auto bufferIter : bufferRange) {
      llvm::dbgs() << "-- " << bufferIter.first->id << " "
                   << bufferIter.first->size << " " << bufferIter.first->offset;
      llvm::dbgs() << " interval " << bufferIter.second.start() << " "
                   << bufferIter.second.end() << "\n";
    }
  }

  void dumpAllocationSize() const {
    LDBG("Dump shared memory allocation size -----------");
    auto liveBuffers = allocation->getLiveBuffers();
    auto analyzedSize = 0;
    for (auto [op, bufferIds] : liveBuffers) {
      auto size = 0;
      for (auto bufferId : bufferIds) {
        auto bufferSize = allocation->getAllocatedSize(bufferId);
        size += bufferSize;
      }
      analyzedSize = std::max(analyzedSize, size);
    }
    llvm::dbgs() << "Allocated: " << allocation->sharedMemorySize
                 << ", analyzed: " << analyzedSize << "\n";
  }

  void dumpInterferenceGraph(const GraphT &interference) const {
    LDBG("\n");
    LDBG("Dump interference graph: \n");
    for (auto edges : interference) {
      llvm::dbgs() << "-- from " << edges.first->id << " to ";
      for (auto node : edges.second) {
        llvm::dbgs() << node->id << "; ";
      }
      llvm::dbgs() << "\n";
    }
  }

  /// Computes the shared memory offsets for all related values.
  /// Paper: Algorithms for Compile-Time Memory Optimization
  /// (https://dl.acm.org/doi/pdf/10.5555/314500.315082)
  void computeOffsets() {
    SmallVector<BufferT *> buffers;
    for (auto bufferIter : bufferRange) {
      buffers.emplace_back(bufferIter.first);
    }

    // Sort buffers by size in descending order to reduce the fragmentation
    // on big buffers caused by smaller buffers. Big buffers have a higher
    // chance to overlap with multiple other buffers, and allocating them first
    // (by calculateStarts) ensures a higher chance that they will occupy a
    // standalone smem slot.
    llvm::stable_sort(
        buffers, [&](BufferT *A, BufferT *B) { return A->size > B->size; });

    calculateStarts(buffers);

    // NOTE: The original paper doesn't consider interference between
    // the bumped ranges. Buffers that previously do not interfere with
    // could interfere after offset bumping if their liveness ranges overlap.
    // Therefore, we rerun the interference graph algorithm after bumping so
    // that we regroup the buffers and color them again. Since we always
    // increase the buffer offset and keep reducing conflicts, we will
    // eventually reach a fixed point.
    GraphT interference;
    buildInterferenceGraph(buffers, interference);
    do {
      allocate(buffers, interference);
      buildInterferenceGraph(buffers, interference);
    } while (!interference.empty());

    // SALA post-pass: compact non-aref buffers to overlap with aref region
    if (salaEnabled) {
      // Check if there are aref buffers and non-aref WG buffers
      bool hasAref = false, hasNonArefWG = false;
      for (auto *buf : buffers) {
        if (buf->owner->hasAttr("aref_buffer"))
          hasAref = true;
        else if (!buf->owner->hasAttr("aref_empty_mbarriers") &&
                 !buf->owner->hasAttr("aref_full_mbarriers") &&
                 buf->owner->getParentOfType<nvidia_gpu::WarpGroupOp>())
          hasNonArefWG = true;
      }
      if (hasAref && hasNonArefWG) {
        salaCompact(buffers, interference);
      }
    }

    LLVM_DEBUG(dumpAllocationSize());
  }

  /// Computes the initial shared memory offsets.
  void calculateStarts(const SmallVector<BufferT *> &buffers) {
    //  v = values in shared memory
    //  t = triplet of (size, start, end)
    //  shared memory space
    //  -
    //  |         *******t4
    //  | /|\ v2 inserts t4, t5, and t6
    //  |  |
    //  | ******t5         ************t6
    //  | ^^^^^v2^^^^^^
    //  |  |      *********************t2
    //  | \|/ v2 erases t1
    //  | ******t1 ^^^^^^^^^v1^^^^^^^^^ ************t3
    //  |---------------------------------------------| liveness range
    //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ...
    // If the available triple's range is less than a given buffer range,
    // we won't know if there has been an overlap without using graph coloring.
    // Start -> Liveness Range
    using TripleMapT = std::multimap<size_t, Interval<size_t>>;
    TripleMapT tripleMap;
    tripleMap.insert(std::make_pair(0, Interval<size_t>()));
    SmallVector<BufferT *> xBuffers = buffers;
    while (!xBuffers.empty()) {
      auto tripleIt = tripleMap.begin();
      auto offset = tripleIt->first;
      auto range = tripleIt->second;
      tripleMap.erase(tripleIt);
      auto bufferIt =
          std::find_if(xBuffers.begin(), xBuffers.end(), [&](auto *buffer) {
            auto xRange = bufferRange[buffer];
            bool res = xRange.intersects(range);
            for (const auto &val : tripleMap)
              res = res &&
                    !val.second.intersects(xRange); // only one buffer intersect
            return res;
          });
      if (bufferIt != xBuffers.end()) {
        auto buffer = *bufferIt;
        auto xSize = buffer->size;
        auto xRange = bufferRange.lookup(buffer);
        // TODO(Keren): A buffer's size shouldn't be determined here, have to
        // clean it up
        size_t alignOffset = buffer->setOffsetAligned(offset);
        tripleMap.insert({alignOffset + xSize,
                          Interval{std::max(range.start(), xRange.start()),
                                   std::min(range.end(), xRange.end())}});
        // We could either insert (range.start, xRange.start) or (range.start,
        // xRange.end), both are correct and determine the potential buffer
        // offset, and the graph coloring algorithm will solve the interference,
        // if any
        if (range.start() < xRange.start())
          tripleMap.insert({offset, Interval{range.start(), xRange.end()}});
        if (xRange.end() < range.end())
          tripleMap.insert({offset, Interval{xRange.start(), range.end()}});
        xBuffers.erase(bufferIt);
      }
    }
    LLVM_DEBUG(dumpBuffers());
  }

  /// Builds a graph of all shared memory values. Edges are created between
  /// shared memory values that are overlapping.
  void buildInterferenceGraph(const SmallVector<BufferT *> &buffers,
                              GraphT &interference) {
    // Reset interference graph
    interference.clear();
    for (auto x : buffers) {
      for (auto y : buffers) {
        if (x == y)
          continue;
        auto xStart = x->offset;
        auto yStart = y->offset;
        auto xSize = x->size;
        auto ySize = y->size;
        Interval xSizeRange = {xStart, xStart + xSize};
        Interval ySizeRange = {yStart, yStart + ySize};
        auto xOpRange = bufferRange.lookup(x);
        auto yOpRange = bufferRange.lookup(y);

        // Buffers interfere if their allocation offsets overlap and they are
        // live at the same time.
        if (xOpRange.intersects(yOpRange) &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }

        // Buffers also interfere if their allocation offsets overlap and they
        // exist within regions that may execute simultaneously with respect to
        // each other.
        auto wsx = x->owner->getParentWithTrait<OpTrait::AsyncRegions>();
        auto wsy = y->owner->getParentWithTrait<OpTrait::AsyncRegions>();
        if (wsx && wsy && wsx == wsy &&
            x->owner->getParentRegion() != y->owner->getParentRegion() &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }

        // Similarly to above, handle interferences among WarpGroupOp regions.
        // This includes cases where one of the buffers is defined at the
        // top level.
        auto wgx = x->owner->getParentOfType<nvidia_gpu::WarpGroupOp>();
        auto wgy = y->owner->getParentOfType<nvidia_gpu::WarpGroupOp>();
        if ((wgx || wgy) && wgx != wgy && xSizeRange.intersects(ySizeRange)) {
          // SALA: skip interference between aref pipeline buffers and
          // non-aref output buffers in different warp groups.
          bool xIsAref = x->owner->hasAttr("aref_buffer") ||
                         x->owner->hasAttr("aref_empty_mbarriers") ||
                         x->owner->hasAttr("aref_full_mbarriers");
          bool yIsAref = y->owner->hasAttr("aref_buffer") ||
                         y->owner->hasAttr("aref_empty_mbarriers") ||
                         y->owner->hasAttr("aref_full_mbarriers");
          // Skip when one is aref (possibly top-level) and the other
          // is a non-aref buffer inside a WarpGroupOp
          if (salaEnabled && xIsAref != yIsAref && wgx != wgy) {
            salaSkippedEdges.insert({x, y});
            continue;
          }
          interference[x].insert(y);
        }
      }
    }

    LLVM_DEBUG(dumpInterferenceGraph(interference));
  }

  /// Finalizes shared memory offsets considering interference.
  void allocate(const SmallVector<BufferT *> &buffers,
                const GraphT &interference) {
    // Reset shared memory size
    allocation->sharedMemorySize = 0;
    // First-fit graph coloring
    // Neighbors are nodes that interfere with each other.
    // We color a node by finding the index of the first available
    // non-neighboring node or the first neighboring node without any color.
    // Nodes with the same color do not interfere with each other.
    DenseMap<BufferT *, int> colors;
    for (auto value : buffers) {
      colors[value] = (value == buffers[0]) ? 0 : -1;
    }
    SmallVector<bool> available(buffers.size());
    for (auto x : buffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto y : interference.lookup(x)) {
        int color = colors[y];
        if (color >= 0) {
          available[color] = false;
        }
      }
      auto it = std::find(available.begin(), available.end(), true);
      colors[x] = std::distance(available.begin(), it);
      LLVM_DEBUG({
        llvm::dbgs() << "-- color " << x->id << " " << colors[x] << "\n";
      });
    }
    // Finalize allocation
    // color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
    // color1: [7, 9) -> [0 + 1 * 15, 9 + 1 * 15) -> [15, 24)
    // color2: [8, 12) -> [8 + 2 * 15, 12 + 2 * 15) -> [38, 42)
    // TODO(Keren): We are wasting memory here.
    // Nodes with color2 can actually start with 24.
    for (auto x : buffers) {
      size_t newOffset = 0;
      for (auto y : interference.lookup(x)) {
        newOffset = std::max(newOffset, y->offset + y->size);
      }
      if (colors.lookup(x) != 0)
        x->setOffsetAligned(newOffset);
      allocation->sharedMemorySize =
          std::max(allocation->sharedMemorySize, x->offset + x->size);
    }
    LLVM_DEBUG(dumpBuffers());
  }

  /// SALA: compact non-aref buffers to offset 0 (overlapping with aref
  /// pipeline region) and re-place barrier buffers tightly.
  void salaCompact(const SmallVector<BufferT *> &buffers,
                   const GraphT &interference) {
    size_t origSize = allocation->sharedMemorySize;

    // Capture baseline offsets for diagnostic
    DenseMap<BufferT *, size_t> baselineOffsets;
    for (auto *buf : buffers)
      baselineOffsets[buf] = buf->offset;

    bool compacted = false;
    for (auto *buf : buffers) {
      if (buf->owner->hasAttr("aref_buffer") ||
          buf->owner->hasAttr("aref_empty_mbarriers") ||
          buf->owner->hasAttr("aref_full_mbarriers"))
        continue;
      if (!buf->owner->getParentOfType<nvidia_gpu::WarpGroupOp>())
        continue;
      // SALA: the happens-before relationship from the aref protocol
      // guarantees that pipeline buffers and the output buffer are
      // never live simultaneously within a single tile iteration.
      buf->setOffsetAligned(0);
      compacted = true;
    }
    if (!compacted)
      return;

    // Re-pack aref pipeline buffers tightly (close gaps left by moved
    // non-aref buffers that previously sat between pipeline stages).
    SmallVector<BufferT *> arefBufs;
    for (auto *buf : buffers)
      if (buf->owner->hasAttr("aref_buffer"))
        arefBufs.push_back(buf);
    llvm::sort(arefBufs,
               [](BufferT *a, BufferT *b) { return a->offset < b->offset; });
    size_t pipelineCursor = 0;
    for (auto *buf : arefBufs) {
      buf->setOffsetAligned(pipelineCursor);
      pipelineCursor = buf->offset + buf->size;
    }

    // Re-place barrier buffers right after the aref pipeline buffers
    size_t pipelineEnd = pipelineCursor;
    for (auto *buf : buffers) {
      if (buf->owner->hasAttr("aref_empty_mbarriers") ||
          buf->owner->hasAttr("aref_full_mbarriers")) {
        buf->setOffsetAligned(pipelineEnd);
        pipelineEnd = buf->offset + buf->size;
      }
    }

    // Recompute total
    allocation->sharedMemorySize = 0;
    for (auto *buf : buffers) {
      allocation->sharedMemorySize =
          std::max(allocation->sharedMemorySize, buf->offset + buf->size);
    }

    // Insert cross-tile barrier for persistent kernels (unless disabled)
    if (!salaNoBarrier)
      salaInsertCrossTileBarrier();

    // Dump IGraph diagnostics
    if (salaDumpIGraph)
      salaDumpIGraphDiagnostic(buffers, interference, origSize,
                               baselineOffsets);

    llvm::errs() << "[SALA] Compaction: " << origSize << " -> "
                 << allocation->sharedMemorySize << " bytes ("
                 << origSize / 1024 << " KB -> "
                 << allocation->sharedMemorySize / 1024 << " KB, "
                 << (100 - 100 * allocation->sharedMemorySize / origSize)
                 << "% reduction)\n";
  }

  /// SALA: total threads in the CTA, derived from the warp-group layout
  /// (max start_warp + num_warps over the module's ttng.warp_group ops), or
  /// -1 when the module has no warp groups.  This is the quantity the
  /// cross-tile barrier's arrival count must equal, so it is what the
  /// geometry guard below checks.
  int salaCtaThreads(Operation *op) {
    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return -1;
    int warps = 0;
    bool found = false;
    mod->walk([&](nvidia_gpu::WarpGroupOp wg) {
      found = true;
      int end = static_cast<int>(wg.getStartWarp()) +
                static_cast<int>(wg.getNumWarps());
      warps = std::max(warps, end);
    });
    return found ? warps * 32 : -1;
  }

  /// SALA: insert cross-tile synchronization for persistent kernels.
  /// Only the TMA load group gets bar.sync; the MMA group proceeds freely.
  /// Correctness relies on the aref protocol's within-tile barriers: the TMA
  /// load group cannot start loading tile N+1 until the MMA group's epilogue
  /// store for tile N is visible. bar.sync in the TMA group's outer loop
  /// ensures this -- the MMA group doesn't need to participate because its
  /// arrive_barrier(empty) after WGMMA already guarantees the store is issued
  /// before the next tile's loads can begin.
  void salaInsertCrossTileBarrier() {
    operation->walk([&](scf::ForOp forOp) {
      auto *parentOp = forOp->getParentOp();
      if (!isa<nvidia_gpu::WarpGroupOp>(parentOp))
        return;

      // Only persistent kernels: must have nested for-loop (K-loop)
      bool hasNestedFor = false;
      forOp.walk([&](scf::ForOp innerFor) {
        if (innerFor != forOp)
          hasNestedFor = true;
      });
      if (!hasNestedFor)
        return;

      // Hard guard: the barrier must be reached by every thread of the CTA, and
      // this prototype only models the 2 x 128-thread geometry.  Any other warp
      // count leaves the barrier half-reached (the kernel deadlocks), so an
      // unsupported geometry is rejected here instead of being compiled.
      int threads = salaCtaThreads(forOp);
      if (threads != 256) {
        std::string got =
            threads > 0 ? ("has " + std::to_string(threads) + " threads")
                        : std::string("has an unknown thread count");
        llvm::report_fatal_error(
            llvm::Twine(
                "SALA: unsupported warp geometry for the cross-tile barrier. "
                "The prototype supports 256-thread kernels (num_warps=4, two "
                "warp groups of 128 threads); this kernel ") +
            llvm::Twine(got) +
            ". Recompile with num_warps=4, or disable SALA (SALA_ENABLE=0).");
      }

      auto loc = forOp.getLoc();
      OpBuilder builder(forOp.getContext());
      auto i32Ty = builder.getIntegerType(32);

      // Insert CTA-wide barrier before scf.yield of the outer tile loop
      auto &yieldOp = forOp.getBody()->back();
      builder.setInsertionPoint(&yieldOp);
      auto barId = builder.create<arith::ConstantIntOp>(loc, 3, i32Ty);
      auto numThreads =
          builder.create<arith::ConstantIntOp>(loc, threads, i32Ty);
      builder.create<NVVM::BarrierOp>(loc, barId, numThreads);
    });
  }

  /// SALA: dump interference graph diagnostics (before and after)
  void salaDumpIGraphDiagnostic(
      const SmallVector<BufferT *> &buffers, const GraphT &interference,
      size_t origSize,
      const DenseMap<BufferT *, size_t> &baselineOffsets) {
    auto bufLabel = [](BufferT *buf) -> std::string {
      std::string label;
      if (buf->owner->hasAttr("aref_buffer"))
        label = "aref_buffer";
      else if (buf->owner->hasAttr("aref_empty_mbarriers"))
        label = "empty_mbar";
      else if (buf->owner->hasAttr("aref_full_mbarriers"))
        label = "full_mbar";
      else if (buf->owner->getParentOfType<nvidia_gpu::WarpGroupOp>())
        label = "output(WG)";
      else
        label = "other";
      return label;
    };

    llvm::errs() << "\n=== SALA Interference Graph Diagnostic ===\n";
    llvm::errs() << "\nBuffers:\n";
    for (auto *buf : buffers) {
      llvm::errs() << "  buf " << buf->id << ": " << buf->size << " bytes, "
                   << bufLabel(buf) << "\n";
    }

    llvm::errs() << "\nOffsets (conventional allocator):\n";
    for (auto *buf : buffers) {
      auto it = baselineOffsets.find(buf);
      size_t off = (it != baselineOffsets.end()) ? it->second : buf->offset;
      llvm::errs() << "  buf " << buf->id << " (" << bufLabel(buf)
                   << "): offset=" << off << "\n";
    }

    llvm::errs() << "\nOffsets (SALA, after compaction):\n";
    for (auto *buf : buffers) {
      llvm::errs() << "  buf " << buf->id << " (" << bufLabel(buf)
                   << "): offset=" << buf->offset << "\n";
    }

    llvm::errs() << "\nSMEM: " << origSize << " -> "
                 << allocation->sharedMemorySize << " bytes ("
                 << origSize / 1024 << " KB -> "
                 << allocation->sharedMemorySize / 1024 << " KB, -"
                 << (100 - 100 * allocation->sharedMemorySize / origSize)
                 << "%)\n";
    llvm::errs() << "===========================================\n\n";
  }

private:
  Operation *operation;
  Allocation::FuncAllocMapT *funcAllocMap;
  Allocation *allocation;
  BufferRangeMapT bufferRange;
  AllocationAnalysisScratchSizeFn scratchSizeGetter;
};

} // namespace triton

void Allocation::run(
    FuncAllocMapT &funcAllocMap,
    triton::AllocationAnalysisScratchSizeFn scratchSizeGetter) {
  triton::AllocationAnalysis(getOperation(), &funcAllocMap, this,
                             scratchSizeGetter);
}

std::map<Operation *, SmallVector<Allocation::BufferId>>
Allocation::getLiveBuffers() {
  std::map<Operation *, SmallVector<BufferId>> liveBuffers;

  Operation *rootOperation = getOperation();
  Liveness liveness(rootOperation);
  auto analyzeOperation = [&](Operation *op) -> void {
    auto scratchBuffer = getBufferId(op);
    if (scratchBuffer != InvalidBufferId)
      liveBuffers[op].push_back(scratchBuffer);
    for (auto result : op->getOpResults()) {
      auto bufferId = getBufferId(result);
      if (bufferId == Allocation::InvalidBufferId)
        continue;
      auto liveOperations = liveness.resolveLiveness(result);
      for (auto depOp : liveOperations)
        liveBuffers[depOp].push_back(bufferId);
    }
  };
  rootOperation->walk(analyzeOperation);
  return liveBuffers;
}

} // namespace mlir
