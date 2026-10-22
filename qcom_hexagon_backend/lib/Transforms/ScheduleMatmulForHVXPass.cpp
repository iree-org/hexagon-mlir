//===- ScheduleMatmulForHVXPass.cpp - Scheduling matmul ops          ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file specifies the rules to schedule matmul op and its variants.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/LinalgUtils.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "schedule-matmul-for-hvx"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_SCHEDULEMATMULFORHVX
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Transposed matmul variants are represented by indexing maps on matmul ops.
static NamedAttribute
getFoldedTransposeIndexingMaps(OpBuilder &builder, linalg::LinalgOp op,
                               linalg::TransposeOp transpose,
                               unsigned operandNumber) {
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  SmallVector<unsigned> permutation(transpose.getPermutation().begin(),
                                    transpose.getPermutation().end());
  AffineMap transposeMap =
      AffineMap::getPermutationMap(permutation, builder.getContext());
  maps[operandNumber] =
      inversePermutation(transposeMap).compose(maps[operandNumber]);
  return builder.getNamedAttr("indexing_maps",
                              builder.getAffineMapArrayAttr(maps));
}

struct ScheduleMatmulForHVXPass
    : public ::impl::ScheduleMatmulForHVXBase<ScheduleMatmulForHVXPass> {
  void runOnOperation() override;
  FailureOr<linalg::GenericOp> GeneralizeOp(IRRewriter &rewriter,
                                            linalg::LinalgOp linalgOp);
};

FailureOr<linalg::GenericOp>
ScheduleMatmulForHVXPass::GeneralizeOp(IRRewriter &rewriter,
                                       linalg::LinalgOp linalgOp) {
  rewriter.setInsertionPoint(linalgOp);
  FailureOr<linalg::LinalgOp> generalizedOp =
      linalg::generalizeNamedOp(rewriter, linalgOp);
  if (failed(generalizedOp)) {
    linalgOp->emitOpError("failed to generalize linalg named op");
    signalPassFailure();
    return failure(); // Return a failure result
  }
  return cast<linalg::GenericOp>(generalizedOp->getOperation());
}

void ScheduleMatmulForHVXPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  // Fold operand transposes into matmul and batch_matmul indexing maps.
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa_and_nonnull<linalg::MatmulOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      auto firstOperandDef = linalgOp.getDpsInputs()[0].getDefiningOp();
      auto secondOperandDef = linalgOp.getDpsInputs()[1].getDefiningOp();
      rewriter.setInsertionPoint(linalgOp);
      if (secondOperandDef && isa<linalg::TransposeOp>(secondOperandDef)) {
        // Check if the second operand is a transpose op
        // Convert linalg.transpose + linalg.matmul to linalg.matmul with
        // transposed B indexing
        auto transposeOp = dyn_cast<linalg::TransposeOp>(secondOperandDef);
        auto matmulTransposeBOp = rewriter.create<linalg::MatmulOp>(
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{linalgOp.getDpsInputs()[0], transposeOp.getOperand(0)},
            linalgOp.getDpsInits(),
            ArrayRef<NamedAttribute>{getFoldedTransposeIndexingMaps(
                rewriter, linalgOp, transposeOp, 1)});
        rewriter.replaceOp(linalgOp, matmulTransposeBOp);
        rewriter.eraseOp(transposeOp);
      } else if (firstOperandDef && isa<linalg::TransposeOp>(firstOperandDef)) {
        // Check if the first operand is a transpose op
        // Convert linalg.transpose + linalg.matmul to linalg.matmul with
        // transposed A indexing
        auto transposeOp = dyn_cast<linalg::TransposeOp>(firstOperandDef);
        auto matmulTransposeAOp = rewriter.create<linalg::MatmulOp>(
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{transposeOp.getOperand(0), linalgOp.getDpsInputs()[1]},
            linalgOp.getDpsInits(),
            ArrayRef<NamedAttribute>{getFoldedTransposeIndexingMaps(
                rewriter, linalgOp, transposeOp, 0)});
        rewriter.replaceOp(linalgOp, matmulTransposeAOp);
        rewriter.eraseOp(transposeOp);
      }
    } else if (isa_and_nonnull<linalg::BatchMatmulOp>(
                   linalgOp.getOperation()) &&
               !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      auto firstOperandDef = linalgOp.getDpsInputs()[0].getDefiningOp();
      auto secondOperandDef = linalgOp.getDpsInputs()[1].getDefiningOp();
      rewriter.setInsertionPoint(linalgOp);
      if (secondOperandDef && isa<linalg::TransposeOp>(secondOperandDef)) {
        // Check if the second operand is a transpose op
        // Convert linalg.transpose + linalg.batch_matmul to
        // linalg.batch_matmul with transposed B indexing op
        auto transposeOp = dyn_cast<linalg::TransposeOp>(secondOperandDef);
        auto matmulTransposeBOp = rewriter.create<linalg::BatchMatmulOp>(
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{linalgOp.getDpsInputs()[0], transposeOp.getOperand(0)},
            linalgOp.getDpsInits(),
            ArrayRef<NamedAttribute>{getFoldedTransposeIndexingMaps(
                rewriter, linalgOp, transposeOp, 1)});
        rewriter.replaceOp(linalgOp, matmulTransposeBOp);
        rewriter.eraseOp(transposeOp);
      } else if (firstOperandDef && isa<linalg::TransposeOp>(firstOperandDef)) {
        // Check if the first operand is a transpose op
        // Convert linalg.transpose + linalg.batch_matmul to
        // linalg.batch_matmul with transposed A indexing op
        auto transposeOp = dyn_cast<linalg::TransposeOp>(firstOperandDef);
        auto matmulTransposeAOp = rewriter.create<linalg::BatchMatmulOp>(
            linalgOp.getLoc(), linalgOp.getOperation()->getResultTypes(),
            ValueRange{transposeOp.getOperand(0), linalgOp.getDpsInputs()[1]},
            linalgOp.getDpsInits(),
            ArrayRef<NamedAttribute>{getFoldedTransposeIndexingMaps(
                rewriter, linalgOp, transposeOp, 0)});
        rewriter.replaceOp(linalgOp, matmulTransposeAOp);
        rewriter.eraseOp(transposeOp);
      }
    }
  });

  SmallVector<linalg::LinalgOp> batchMatmulOps;
  // Collect all linalg::BatchMatmulOps that do not have the
  // "library_call" attribute
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa_and_nonnull<linalg::BatchMatmulOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      batchMatmulOps.push_back(linalgOp);
    }
  });
  for (auto linalgOp : batchMatmulOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    auto permutation = getBatchMatmulPermutation(linalgOp);
    if (permutation.empty()) {
      linalgOp->emitOpError("failed to determine batch matmul permutation");
      return signalPassFailure();
    }
    rewriter.setInsertionPoint(*generalizedOp);
    // Apply the interchange transformation
    FailureOr<linalg::GenericOp> interchangedOp =
        linalg::interchangeGenericOp(rewriter, *generalizedOp, permutation);
    if (failed(interchangedOp)) {
      generalizedOp->emitOpError(
          "failed to apply interchange to linalg.generic op");
      return signalPassFailure();
    }
  }

  SmallVector<linalg::LinalgOp> matmulOps;
  // Collect all linalg::MatmulOps that do not have the "library_call" attribute
  funcOp.walk([&](linalg::MatmulOp linalgOp) {
    if (isa_and_nonnull<linalg::LinalgOp>(linalgOp.getOperation()) &&
        !linalgOp->getAttrOfType<StringAttr>("library_call")) {
      matmulOps.push_back(linalgOp);
    }
  });
  for (auto linalgOp : matmulOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    auto permutation = getMatmulPermutation(linalgOp);
    if (permutation.empty()) {
      linalgOp->emitOpError("failed to determine matmul permutation");
      return signalPassFailure();
    }
    rewriter.setInsertionPoint(*generalizedOp);
    // Apply the interchange transformation
    FailureOr<linalg::GenericOp> interchangedOp =
        linalg::interchangeGenericOp(rewriter, *generalizedOp, permutation);
    if (failed(interchangedOp)) {
      generalizedOp->emitOpError(
          "failed to apply interchange to linalg.generic op");
      return signalPassFailure();
    }
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createScheduleMatmulForHVXPass() {
  return std::make_unique<ScheduleMatmulForHVXPass>();
}
