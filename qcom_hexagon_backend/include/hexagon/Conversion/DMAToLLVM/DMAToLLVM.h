//===- DMAToLLVM.h - DMA to LLVM pass ---------------------------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_DMATOLLVM_DMATOLLVM_H
#define HEXAGON_CONVERSION_DMATOLLVM_DMATOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace hexagon {

#define GEN_PASS_DECL
#include "hexagon/Conversion/DMAToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createDMAToLLVMPass();

/// Populates the DMA-to-LLVM patterns into a caller-owned conversion.
/// The caller is responsible for configuring the conversion target and type
/// converter, including Hexagon address-space conversions.
void populateDMAToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_DMATOLLVM_DMATOLLVM_H
