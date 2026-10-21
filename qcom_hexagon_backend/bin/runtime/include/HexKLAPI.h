//===- HexKLAPI.h - HExKL C runtime call API   -----------------------------==//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HEXKLAPI_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_HEXKLAPI_H

#include "HAP_power.h"
#include <memory>
#include <stdint.h>

extern "C" {
int hexkl_matmul_f16f16_f32(uint64_t n_row, uint64_t n_col, uint64_t n_inner,
                            float *restrict outM,
                            const _Float16 *restrict inAct,
                            const _Float16 *restrict inW);
}

extern "C" {
/// Bytes of VTCM scratch hexkl_matmul_f16f16_f32_with_scratch needs.
size_t hexkl_matmul_f16f16_f32_scratch_bytes(uint64_t n_row, uint64_t n_col,
                                             uint64_t n_inner);

/// hexkl_matmul_f16f16_f32 with caller-provided VTCM scratch of at least
/// hexkl_matmul_f16f16_f32_scratch_bytes() bytes, aligned to
/// HEXKL_HMX_ACTIVATION_ALIGNMENT (2048) bytes.
int hexkl_matmul_f16f16_f32_with_scratch(uint64_t n_row, uint64_t n_col,
                                         uint64_t n_inner, float *restrict outM,
                                         const _Float16 *restrict inAct,
                                         const _Float16 *restrict inW,
                                         void *vtcm_scratch);
}
#endif // # HEXAGON_BIN_RUNTIME_INCLUDE_HEXKLAPI_H
