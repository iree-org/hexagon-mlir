//===-- DecomposeHexKLMatmulPass.cpp - Decompose hexkl.matmul to micro ops ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to decompose hexkl::MatmulOp into hexkl micro HMX operations.
// This pass implements the tiling strategy from hexkl_matmul_f16f16_f32.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/HexKLToLLVM/HexKLExternalFnNames.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "decompose-hexkl-matmul"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_DECOMPOSEHEXKLMATMUL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Fork divergence note: upstream modeled the scratch/config layout as fixed
// 4096-byte regions with a hardcoded extra reserve. We modified it to use
// 2048-byte fp16 tile spacing and place the HMX configuration block at the end
// of the VTCM allocation, with the exact size reported by
// `hexkl_micro_hmx_config_size()`. Materializing the declaration here keeps the
// decomposition pass aligned with hexKLAPI.cpp and with the later HexKL-to-LLVM
// lowering, which computes the config offset as `vtcm_size - config_size`.
// I am unsure if this divergence was intentional in the original upstream
// version
static func::FuncOp getOrCreateHmxConfigSizeFunc(ModuleOp module, Location loc,
                                                 PatternRewriter &rewriter) {
  const std::string fnName = hexkl::getHmxConfigSizeFnName();
  if (auto funcOp = module.lookupSymbol<func::FuncOp>(fnName)) {
    return funcOp;
  }

  auto savedInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  auto funcType = rewriter.getFunctionType({}, rewriter.getI32Type());
  auto funcOp = rewriter.create<func::FuncOp>(loc, fnName, funcType);
  funcOp.setPrivate();
  rewriter.restoreInsertionPoint(savedInsertionPoint);
  return funcOp;
}

struct DecomposeHexKLMatmul final : public OpRewritePattern<hexkl::MatmulOp> {
  DecomposeHexKLMatmul(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(hexkl::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value result = op.getOuts();

    auto lhsType = cast<MemRefType>(lhs.getType());
    auto rhsType = cast<MemRefType>(rhs.getType());
    auto resultType = cast<MemRefType>(result.getType());

    // Validate rank (must be 2D)
    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    if (lhsShape.size() != 2 || rhsShape.size() != 2 ||
        resultShape.size() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matmul supported");
    }

    // Validate static shape compatibility if available
    if (lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
        resultType.hasStaticShape()) {
      int64_t M = lhsShape[0];
      int64_t K_lhs = lhsShape[1];
      int64_t K_rhs = rhsShape[0];
      int64_t N = rhsShape[1];
      int64_t M_out = resultShape[0];
      int64_t N_out = resultShape[1];

      // Validate dimensions match: lhs(M×K) × rhs(K×N) = result(M×N)
      if (K_lhs != K_rhs) {
        return rewriter.notifyMatchFailure(
            op, "inner dimensions mismatch: lhs K != rhs K");
      }
      if (M != M_out || N != N_out) {
        return rewriter.notifyMatchFailure(op, "output dimensions mismatch");
      }
    }

    // Create constants
    auto i32Ty = rewriter.getI32Type();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    Value idx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value idx1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value idx32 = rewriter.create<arith::ConstantIndexOp>(loc, 32);
    Value idx31 = rewriter.create<arith::ConstantIndexOp>(loc, 31);

    Value i32_0 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 0);
    Value i32_1 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 1);
    Value i32_32 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 32);
    Value i32_2048 = rewriter.create<arith::ConstantIntOp>(loc, i32Ty, 2048);
    Value idx2048 = rewriter.create<arith::ConstantIndexOp>(loc, 2048);

    // Get dimensions dynamically
    Value dimM = rewriter.create<memref::DimOp>(loc, lhs, idx0);
    Value dimK = rewriter.create<memref::DimOp>(loc, lhs, idx1);
    Value dimN = rewriter.create<memref::DimOp>(loc, rhs, idx1);

    Value M = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimM);
    Value K = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimK);
    Value N = rewriter.create<arith::IndexCastOp>(loc, i32Ty, dimN);

    // Calculate numKTiles = (k + 31) / 32
    Value kPlus31 = rewriter.create<arith::AddIOp>(loc, dimK, idx31);
    Value kTiles = rewriter.create<arith::DivUIOp>(loc, kPlus31, idx32);
    Value kTilesI32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, kTiles);

    // Fork divergence note: keep this layout in sync with the HexKL micro
    // runtime rather than upstream's older fixed-size approximation. The
    // runtime uses 2048-byte fp16 tile regions and appends a trailing HMX
    // config block whose size comes from `hexkl_micro_hmx_config_size()`.
    // Reserving that exact suffix here avoids placing the config area inside
    // the tile data region after HexKL-to-LLVM lowers
    // `MicroHMXSetupAccReadF16Op` / `MicroHMXAccReadF16Op` to runtime calls
    // that derive the config offset from the total VTCM size.

    // Calculate the fp16 VTCM data layout to match the reference HexKL API:
    // [act_tiles | scratch_tiles | flat_out | acc_read] + config_bytes
    // with each fp16 tile region 2048-byte aligned.
    Value twoKTiles = rewriter.create<arith::MulIOp>(
        loc, kTiles, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value dataTiles = rewriter.create<arith::AddIOp>(
        loc, twoKTiles, rewriter.create<arith::ConstantIndexOp>(loc, 2));
    Value dataBytes = rewriter.create<arith::MulIOp>(loc, dataTiles, idx2048);
    auto cfgSizeFunc = getOrCreateHmxConfigSizeFunc(module, loc, rewriter);
    Value cfgSizeI32 =
        rewriter.create<func::CallOp>(loc, cfgSizeFunc, ValueRange{})
            .getResult(0);
    Value cfgSize = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), cfgSizeI32);
    Value vtcmBytes = rewriter.create<arith::AddIOp>(loc, dataBytes, cfgSize);

    // Manual VTCM alloc; tag for bufferization and dealloc after outer loop.
    auto vtcmType =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                        MemRefLayoutAttrInterface{},
                        IntegerAttr::get(rewriter.getI32Type(), 1));
    auto vtcmAlloc =
        rewriter.create<hexagonmem::AllocOp>(loc, vtcmType, vtcmBytes);
    vtcmAlloc->setAttr("bufferization.manual_deallocation",
                       rewriter.getUnitAttr());
    Value vtcm = vtcmAlloc.getResult();

    // Setup HMX accumulator
    rewriter.create<hexkl::MicroHMXSetupAccReadF16Op>(loc, vtcm);

    // Hoist loop-invariant offset calculations
    // weightOffset = numKTiles * 2048 (weight buffer starts after activation
    // tiles)
    Value wOff = rewriter.create<arith::MulIOp>(loc, kTilesI32, i32_2048);

    // flatOffset = numKTiles * 2048 (flat output buffer location)
    Value flatOff = wOff;

    // accReadOffset = (numKTiles + 1) * 2048 (accumulator readback location)
    Value kTilesPlus1 = rewriter.create<arith::AddIOp>(loc, kTilesI32, i32_1);
    Value accOff = rewriter.create<arith::MulIOp>(loc, kTilesPlus1, i32_2048);

    // Outer loop: iterate over rows (M dimension) in 32-row tiles
    auto outerFor = rewriter.create<scf::ForOp>(
        loc, idx0, dimM, idx32, ValueRange{},
        [&](OpBuilder &b, Location loc, Value row, ValueRange) {
          Value rowI32 = b.create<arith::IndexCastOp>(loc, i32Ty, row);
          Value rowTile = b.create<arith::DivUIOp>(loc, rowI32, i32_32);

          // Load and layout activation tiles for this row
          b.create<scf::ForOp>(
              loc, idx0, kTiles, idx1, ValueRange{},
              [&](OpBuilder &bb, Location loc, Value ktIdx, ValueRange) {
                Value kt = bb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);

                // scratchOffset = (numKTiles + kTile) * 2048
                Value scrIdx = bb.create<arith::AddIOp>(loc, kTilesI32, kt);
                Value scrOff = bb.create<arith::MulIOp>(loc, scrIdx, i32_2048);

                // Copy activation tile to scratch
                bb.create<hexkl::MicroHMXCopySubmatrixToF16Op>(
                    loc, vtcm, scrOff, lhs, rowTile, kt, M, K);

                // Layout activation from scratch to activation buffer
                Value actOff = bb.create<arith::MulIOp>(loc, kt, i32_2048);
                bb.create<hexkl::MicroHMXRmToAhF16Op>(loc, vtcm, actOff,
                                                      scrOff);

                bb.create<scf::YieldOp>(loc);
              });

          // Inner loop: iterate over columns (N dimension) in 32-col tiles
          b.create<scf::ForOp>(
              loc, idx0, dimN, idx32, ValueRange{},
              [&](OpBuilder &bb, Location loc, Value col, ValueRange) {
                Value colI32 = bb.create<arith::IndexCastOp>(loc, i32Ty, col);
                Value colTile = bb.create<arith::DivUIOp>(loc, colI32, i32_32);

                // Clear accumulator
                bb.create<hexkl::MicroHMXAccClearF16Op>(loc);

                // Innermost loop: iterate over K tiles for accumulation
                bb.create<scf::ForOp>(
                    loc, idx0, kTiles, idx1, ValueRange{},
                    [&](OpBuilder &bbb, Location loc, Value ktIdx, ValueRange) {
                      Value kt =
                          bbb.create<arith::IndexCastOp>(loc, i32Ty, ktIdx);

                      // Load weight tile
                      bbb.create<hexkl::MicroHMXRmToWhF16Op>(
                          loc, vtcm, wOff, rhs, kt, colTile, N);

                      // Perform matrix multiplication (accumulates)
                      Value actOff2 =
                          bbb.create<arith::MulIOp>(loc, kt, i32_2048);
                      bbb.create<hexkl::MicroHMXMmF16Op>(loc, vtcm, actOff2,
                                                         wOff);

                      bbb.create<scf::YieldOp>(loc);
                    });

                // Read accumulator
                bb.create<hexkl::MicroHMXAccReadF16Op>(loc, vtcm, accOff);

                // Convert from activation layout to flat layout
                bb.create<hexkl::MicroHMXAhToRmF16Op>(loc, vtcm, flatOff,
                                                      accOff);

                // Copy result to output (f16 -> f32)
                bb.create<hexkl::MicroHMXCopyF16ToF32SubmatrixOp>(
                    loc, vtcm, flatOff, result, rowTile, colTile, M, N);

                bb.create<scf::YieldOp>(loc);
              });

          b.create<scf::YieldOp>(loc);
        });

    // Explicitly deallocate VTCM buffer to avoid relying on ConvertToHexagonmem
    // rewriting of generic memref.dealloc for dynamic VTCM types.
    rewriter.setInsertionPointAfter(outerFor);
    rewriter.create<hexagonmem::DeallocOp>(loc, vtcm);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateDecomposeHexKLMatmulPatterns(RewritePatternSet &patterns) {
  patterns.add<DecomposeHexKLMatmul>(patterns.getContext());
}

struct DecomposeHexKLMatmulPass
    : public ::impl::DecomposeHexKLMatmulBase<DecomposeHexKLMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<hexkl::HexKLDialect, hexagonmem::HexagonMemDialect,
                arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateDecomposeHexKLMatmulPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createDecomposeHexKLMatmulPass() {
  return std::make_unique<DecomposeHexKLMatmulPass>();
}
