// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <optional>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_executor.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#if defined(OV_CPU_WITH_MLAS) && defined(OPENVINO_ARCH_X86_64)
#    include "nodes/executors/mlas/mlas_gemm.hpp"
#endif
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <common/memory_desc_wrapper.hpp>

#    include "cpu_types.h"
#    include "memory_desc/cpu_memory_desc_utils.h"
#    include "memory_desc/dnnl_memory_desc.h"
#    include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#    include "onednn/iml_type_mapper.h"
#endif

#if defined(OV_CPU_WITH_KLEIDIAI)
#    include "nodes/executors/kleidiai/kleidiai_mm.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_fullyconnected.hpp"
#    include "nodes/executors/acl/acl_lowp_fullyconnected.hpp"
#    include "nodes/executors/common/common_utils.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig dnnlFCLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};
static const LayoutConfig aclFCLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};

/**
 * @brief Creates an AB8b24a blocked memory descriptor for transformer weight matrices
 *
 * This format is optimized for AMD Ryzen AVX2 BRGEMM micro-kernels processing quantized INT8 weights.
 * The 24×8 tile size matches the hardcoded BRGEMM kernel requirements and AVX2 SIMD width.
 *
 * Format structure: [Outer_A][Outer_B][inner_b=24][inner_a=8]
 *   - Outer_A = ceil(rows / 24)  // Number of 24-row blocks
 *   - Outer_B = cols / 8         // Number of 8-column blocks
 *   - inner_b = 24               // Rows per block (BRGEMM micro-kernel requirement)
 *   - inner_a = 8                // Columns per block (AVX2 vector width)
 *
 * Memory layout: weight[a][b] = memory[a/24][b/8][a%24][b%8]
 *
 * @param prc Element precision (typically u8 for quantized weights)
 * @param shape Original 2D weight tensor shape [rows, cols]
 * @return Blocked memory descriptor with AB8b24a format
 *
 * Example dimensions:
 *   - Q/K/V projections: 256×1536 → [11][192][24][8] (3.1% padding on rows)
 *   - Attention output:  1536×1536 → [64][192][24][8] (0% padding, perfect fit)
 *   - FFN expand:        8960×1536 → [374][192][24][8] (0.18% padding on rows)
 *   - FFN contract:      1536×8960 → [64][1120][24][8] (0% padding, perfect fit)
 */
static std::shared_ptr<DnnlBlockedMemoryDesc> createAB8b24aDescriptor(ov::element::Type prc, const Shape& shape) {
    OPENVINO_ASSERT(shape.getRank() == 2, "AB8b24a format only applies to 2D tensors");
    
    const auto& dims = shape.getDims();
    const size_t rows = dims[0];  // A dimension
    const size_t cols = dims[1];  // B dimension
    
    // AB8b24a blocking: tiles of 24 rows × 8 columns
    constexpr size_t BLOCK_SIZE_A = 24;  // inner_b: rows per tile
    constexpr size_t BLOCK_SIZE_B = 8;   // inner_a: columns per tile
    
    // Calculate outer block dimensions (with padding if needed)
    const size_t outer_A = (rows + BLOCK_SIZE_A - 1) / BLOCK_SIZE_A;  // ceil(rows / 24)
    const size_t outer_B = cols / BLOCK_SIZE_B;                        // cols must be divisible by 8
    
    OPENVINO_ASSERT(cols % BLOCK_SIZE_B == 0, 
                    "Column dimension ", cols, " must be divisible by ", BLOCK_SIZE_B, 
                    " for AB8b24a format");
    
    // Blocked dimensions: [outer_A, outer_B, inner_b=24, inner_a=8]
    // The memory layout is: [outer_A][outer_B][BLOCK_SIZE_A][BLOCK_SIZE_B]
    VectorDims blockedDims = {
        outer_A,      // Outer blocks in A dimension
        outer_B,      // Outer blocks in B dimension
        BLOCK_SIZE_A, // Inner block size for A (24 rows)
        BLOCK_SIZE_B  // Inner block size for B (8 cols)
    };
    
    // Order vector maps blocked dims to original dims:
    // First 2 elements are the outer dimension order (A=0, B=1)
    // Last 2 elements are the inner block indices (0=A inner, 1=B inner)
    VectorDims order = {
        0,  // outer_A maps to original dim 0 (rows)
        1,  // outer_B maps to original dim 1 (cols)
        0,  // inner block of 24 belongs to dim 0 (rows)
        1   // inner block of 8 belongs to dim 1 (cols)
    };
    
    // Strides in descending order (outermost to innermost):
    // stride[0] = outer_B * BLOCK_SIZE_A * BLOCK_SIZE_B  (moving 1 outer_A block)
    // stride[1] = BLOCK_SIZE_A * BLOCK_SIZE_B            (moving 1 outer_B block)
    // stride[2] = BLOCK_SIZE_B                            (moving 1 row within block)
    // stride[3] = 1                                       (moving 1 element)
    VectorDims strides = {
        outer_B * BLOCK_SIZE_A * BLOCK_SIZE_B,  // Stride for outer_A
        BLOCK_SIZE_A * BLOCK_SIZE_B,             // Stride for outer_B
        BLOCK_SIZE_B,                            // Stride for inner_b (row within tile)
        1                                         // Stride for inner_a (element)
    };
    
    // No offset padding or offset to data
    const size_t offsetPadding = 0;
    const VectorDims offsetPaddingToData = {};
    
    return std::make_shared<DnnlBlockedMemoryDesc>(
        prc,
        shape,
        blockedDims,
        order,
        offsetPadding,
        offsetPaddingToData,
        strides
    );
}

/**
 * @brief Custom optimal config creator for FullyConnected operations with transformer-specific optimizations
 *
 * This function checks weight tensor dimensions and declares AB8b24a blocked format preference for
 * critical transformer operations:
 *   - Attention projections (Q/K/V): 256×1536 or 1536×1536
 *   - FFN expand layer: 8960×1536
 *   - FFN contract layer: 1536×8960
 *
 * For other dimension combinations, falls back to standard plain format.
 *
 * Rationale for dimension-specific blocking:
 *   - 1536 dimension: Perfect fit for 24-row blocks (1536÷24=64, zero padding)
 *   - 8960 dimension: Minimal padding overhead (8960÷24=373.33→374, 0.18% padding)
 *   - Eliminates 206.23ms of runtime reorder overhead per 24-block transformer inference
 *
 * @param config Current executor configuration with memory descriptors and attributes
 * @return Optional<Config> with AB8b24a blocked descriptors for weights if dimensions match critical paths,
 *         or standard config otherwise
 */
static std::optional<executor::Config<FCAttrs>> createTransformerOptimalConfig(const FCConfig& config) {
    // Check if weight tensor exists and is 2D
    auto weiDescIt = config.descs.find(ARG_WEI);
    if (weiDescIt == config.descs.end() || weiDescIt->second->empty()) {
        return createOptimalConfigCommon(config, dnnlFCTypeMapping, dnnlFCLayoutConfig, fcMappingNotation);
    }
    
    const auto& weiDesc = weiDescIt->second;
    const auto& weiShape = weiDesc->getShape();
    
    // Only apply blocking to 2D weight tensors (standard FC/MatMul)
    if (weiShape.getRank() != 2) {
        return createOptimalConfigCommon(config, dnnlFCTypeMapping, dnnlFCLayoutConfig, fcMappingNotation);
    }
    
    const auto& weiDims = weiShape.getDims();
    const size_t rows = weiDims[0];
    const size_t cols = weiDims[1];
    
    // Check for transformer critical dimensions:
    // - Dimension 1536: Hidden size (attention operations, FFN contract output)
    // - Dimension 8960: FFN intermediate size (5.833 × hidden_size)
    // - Dimension 256: Attention head dimension (1536 ÷ 6 heads)
    //
    // Critical operation patterns in Qwen2-1.5B:
    //   1. Q/K/V projections: 256×1536 (24 blocks × 3 projections = 72 ops)
    //   2. Attention output:  1536×1536 (24 blocks × 1 projection = 24 ops)
    //   3. FFN expand:        8960×1536 (24 blocks × 1 layer = 24 ops)
    //   4. FFN contract:      1536×8960 (24 blocks × 1 layer = 24 ops)
    const bool isAttentionDim = (rows == 256 || rows == 1536) && cols == 1536;
    const bool isFFNExpandDim = rows == 8960 && cols == 1536;
    const bool isFFNContractDim = rows == 1536 && cols == 8960;
    
    const bool isCriticalPath = isAttentionDim || isFFNExpandDim || isFFNContractDim;
    
    if (!isCriticalPath) {
        // Not a critical transformer operation, use default plain format
        return createOptimalConfigCommon(config, dnnlFCTypeMapping, dnnlFCLayoutConfig, fcMappingNotation);
    }
    
    // Verify column dimension is compatible with 8-element blocking (AVX2 requirement)
    if (cols % 8 != 0) {
        // Column dimension not compatible with AB8b24a, fall back to plain
        return createOptimalConfigCommon(config, dnnlFCTypeMapping, dnnlFCLayoutConfig, fcMappingNotation);
    }
    
    // Get precision mapping from type configuration
    const auto typeConfig = getTypeConfiguration(config.descs, dnnlFCTypeMapping, fcMappingNotation);
    const auto weiType = typeConfig.at(ARG_WEI);
    
    // Create optimal descriptors with AB8b24a blocking for weights
    MemoryDescArgs optimalDescs = config.descs;
    
    // Replace weight descriptor with AB8b24a blocked format
    optimalDescs[ARG_WEI] = createAB8b24aDescriptor(weiType, weiShape);
    
    // Update other descriptors to match type configuration (src, bias, dst remain plain format)
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    
    if (optimalDescs.count(ARG_SRC) && !optimalDescs[ARG_SRC]->empty()) {
        const auto srcType = typeConfig.at(ARG_SRC);
        if (optimalDescs[ARG_SRC]->getPrecision() != srcType) {
            optimalDescs[ARG_SRC] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                srcType, optimalDescs[ARG_SRC]->getShape());
        }
    }
    
    if (optimalDescs.count(ARG_BIAS) && !optimalDescs[ARG_BIAS]->empty()) {
        const auto biasType = typeConfig.at(ARG_BIAS);
        if (optimalDescs[ARG_BIAS]->getPrecision() != biasType) {
            optimalDescs[ARG_BIAS] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                biasType, optimalDescs[ARG_BIAS]->getShape());
        }
    }
    
    if (optimalDescs.count(ARG_DST) && !optimalDescs[ARG_DST]->empty()) {
        const auto dstType = typeConfig.at(ARG_DST);
        if (optimalDescs[ARG_DST]->getPrecision() != dstType) {
            optimalDescs[ARG_DST] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
                dstType, optimalDescs[ARG_DST]->getShape());
        }
    }
    
    return std::optional<executor::Config<FCAttrs>>(executor::Config<FCAttrs>{optimalDescs, config.attrs});
}

template <dnnl::impl::cpu::x64::cpu_isa_t ISA>
struct Require {
    bool operator()() {
        return dnnl::impl::cpu::x64::mayiuse(ISA);
    }
};

// clang-format off
static const TypeMapping dnnlFCTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},               {bypass(), bypass(), use<3>(), bypass()}},
    {{_f16, _f16, _any, _f16 | _f32},                         {bypass(), bypass(), use<3>(), bypass()}},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            {bypass(), bypass(), use<0>(), use<0>()}},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any},                  {bypass(), bypass(), use<0>(), use<0>()}},
    {{_bf16, _f16, _any, _any},                        {bypass(), bypass(), use<0>(), use<0>()}},
    {{_f16, _bf16, _any, _any},                        {bypass(), bypass(), use<0>(), use<0>()}},
    // quantization configuration
    // int8 inner_product does not support f16 output or bias (f16 output is only supported on X86_64 platforms)
#if defined(OPENVINO_ARCH_X86_64)
    {{_u8 | _i8, _i8, _u8 | _i8 | _i32 | _bf16 | _f32 | _dynamic, _u8 | _i8 | _i32 | _bf16 | _f16 | _f32}, {bypass(), bypass(), bypass(),  bypass()}},
#else
    {{_u8 | _i8, _i8, _u8 | _i8 | _i32 | _bf16 | _f32 | _dynamic, _u8 | _i8 | _i32 | _bf16 | _f32}, {bypass(), bypass(), bypass(),  bypass()}},
#endif
    {{_u8 | _i8, _i8, _f16, _u8 | _i8 | _i32 | _bf16 | _f32}, {bypass(), bypass(), just<f32>(), bypass()}},
    {{_u8 | _i8, _i8, _any, _any}, {bypass(), bypass(), just<f32>(), just<f32>()}},
    // compresses int weights (@todo more strict requrements for output precision?)
    {{_bf16, _u8 | _i8 | _nf4 | _u4 | _i4 | _f4e2m1 | _u2, _any, _any},       {bypass(), bypass(), use<0>(), use<0>()},
     Require<dnnl::impl::cpu::x64::avx512_core_bf16>()}, // Ticket 122347
    {{_bf16, _u8 | _i8 | _nf4 | _u4 | _i4 | _f4e2m1, _any, _any},       {just<f32>(), bypass(), just<f32>(), just<f32>()}},
    {{_f32,  _u8 | _i8 | _nf4 | _u4 | _i4 | _f4e2m1 | _u2, _any, _any},       {bypass(), bypass(), use<0>(), use<0>()}},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                {just<f32>(), just<f32>(), just<f32>(), just<f32>()}},
    // @todo explicitly cover configuration limitations for oneDNN on ARM
};

static const TypeMapping aclFCTypeMapping {
    // {src, wei, bia, dst}                  pt<src, wei, bias, dst>
    {{_f32 | _f16, _f32 | _f16, _any, _any}, {bypass(), bypass(), use<0>(), use<0>()}},
    {{_any, _any, _any, _any},               {just<f32>(), just<f32>(), just<f32>(), just<f32>()}}
};

static const TypeMapping aclLowpFCTypeMapping {
    // {src, wei, bia, dst}                  pt<src, wei, bias, dst>
    {{_i8, _i8, _any, _f32},                 {bypass(), bypass(), use<3>(), bypass()}}
};

static const MappingNotation fcMappingNotation {
    {ARG_SRC,  0},
    {ARG_WEI,  1},
    {ARG_BIAS, 2},
    {ARG_DST,  3}
};

static const TypeMapping dnnlConvolutionTypeMapping {
    // {src, wei, bia, dst}                        pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},    {bypass(), bypass(), use<3>(), bypass()}},
    {{_f16, _f16, _any, _f16 | _f32},              {bypass(), bypass(), use<3>(), bypass()}},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8}, {bypass(), bypass(), use<0>(), use<0>()}},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any},       {bypass(), bypass(), use<0>(), use<0>()}},
    {{_bf16, _f16, _any, _any},             {bypass(), bypass(), use<0>(), use<0>()}},
    {{_f16, _bf16, _any, _any},             {bypass(), bypass(), use<0>(), use<0>()}},
    // quantization configuration
    {{_u8 | _i8, _i8, _any, _any},                 {bypass(), bypass(), use<3>(), bypass()}},
    // @todo should we fallback to _fxx instead of _f32 (currenly legacy logic is replicated)
    {{_any, _any, _any, _any},                     {just<f32>(), just<f32>(), just<f32>(), just<f32>()}},
};

static const TypeMapping dnnlMatMulTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},               {bypass(), bypass(), use<3>(), bypass()}},
    {{_f16, _f16, _any, _f16 | _f32},                         {bypass(), bypass(), use<3>(), bypass()}},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            {bypass(), bypass(), use<0>(), use<0>()}},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any},                  {bypass(), bypass(), use<0>(), use<0>()}},
    {{_bf16, _f16, _any, _any},                        {bypass(), bypass(), use<0>(), use<0>()}},
    {{_f16, _bf16, _any, _any},                        {bypass(), bypass(), use<0>(), use<0>()}},
    // quantization configuration
    {{_u8 | _i8, _i8, _u8|_i8|_i32|_bf16|_f16|_f32|_dynamic, _u8|_i8|_i32|_bf16|_f16|_f32}, {bypass(), bypass(), bypass(),  bypass()}},
    {{_u8 | _i8, _i8, _any, _any},                            {bypass(), bypass(), just<f32>(), just<f32>()}},
    // compresses int weights
    {{_f32 | _bf16 | _f16, _u8 | _i8, _any, _any},            {bypass(), bypass(), use<0>(), use<0>()}},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                {just<f32>(), just<f32>(), just<f32>(), just<f32>()}},
    // @todo explicitly cover configuration limitations for oneDNN on ARM
};
// clang-format on

[[maybe_unused]] static inline bool noWeightsDecompression(const FCConfig& config) {
    return !DnnlFCPrimitive::useWeightsDecompressionImpl(srcType(config), weiType(config), config.attrs.modelType);
}

[[maybe_unused]] static inline bool noSparseDecompression(const FCConfig& config) {
    return !(config.attrs.sparseWeights);
}

[[maybe_unused]] static inline bool noPostOps(const FCConfig& config) {
    return config.attrs.postOps.empty();
}

[[maybe_unused]] static inline bool dnnlMatMulSupportedPrecision(const FCConfig& config) {
    // support regular float type matmul
    if (any_of(srcType(config), f32, f16, bf16) && any_of(weiType(config), f32, f16, bf16)) {
        return true;
    }
    // i32 can be up converted to f32
    if (any_of(srcType(config), i32) && any_of(weiType(config), i32)) {
        return true;
    }
    // support integer type quantization matmul
    return any_of(srcType(config), u8, i8) && any_of(weiType(config), u8, i8);
}

struct CreateOptimalConfigDefault {
    std::optional<ConvConfig> operator()(const ConvConfig& config) const {
        return createOptimalConfigCommon(config, dnnlMatMulTypeMapping, dnnlFCLayoutConfig, fcMappingNotation);
    }
};

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<FCAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<FCAttrs>> fullyconnectedImplementations {
        OV_CPU_INSTANCE_MLAS_X64(
            "fullyconnected_mlas",
            ExecutorType::Mlas,
            OperationType::MatMul,
            // supports
            [](const FCConfig& config) -> bool {
                // @todo probably there is no need of having implementation name in the debug message
                // since it can be distinguished from the context of other logs anyway.
                VERIFY(noPostOps(config), UNSUPPORTED_POST_OPS);
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                VERIFY(all_of(f32, srcType(config), weiType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(MlasGemmExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);
                VERIFY(weiRank(config) <= 3U, UNSUPPORTED_WEI_RANK);
                VERIFY(weiRank(config) != 3U || weiDims(config)[0] <= 1, UNSUPPORTED_WEI_RANK);
                return true;
            },
            HasNoOptimalConfig<FCAttrs>{},
            AcceptsAnyShape<FCAttrs>,
            CreateDefault<MlasGemmExecutor, FCAttrs>{}
            )
        OV_CPU_INSTANCE_X64(
            "convolution_1x1_dnnl",
            ExecutorType::Dnnl,
            OperationType::Convolution,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                auto getOffset0 = [](const MemoryDescPtr& desc) {
                    DnnlMemoryDescCPtr dnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(desc);
                    dnnl::impl::memory_desc_wrapper wrapped(dnnlDesc->getDnnlDesc().get());
                    return wrapped.offset0();
                };

                VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);
                VERIFY(srcType(config) == ov::element::f32, UNSUPPORTED_SRC_PRECISIONS);
                // disable rank=4:
                // if layout is nhwc:
                //   A matrix: N * IC * H * W --> N * (IC*H*W), the M, N', K of matrix multiply will be:
                //   M = 1, K = (IC*H*W), when M = 1 it should not be efficient since acts as a vector multiply
                // if layout is nchw/nChw16c: brg1x1 not support. Although jit supports, it should have similar
                //   problems with the above.
                VERIFY(any_of(srcRank(config), 2U, 3U), UNSUPPORTED_SRC_RANK);
                VERIFY(weiRank(config) == 2, UNSUPPORTED_WEI_RANK);
                // brg convolution does not support stride
                VERIFY(getOffset0(config.descs.at(ARG_DST)) == 0, UNSUPPORTED_DST_STRIDES);
                return true;
            },
            // createOptimalConfig
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                // @todo use dnnlConvolutionLayoutConfig after one is implemented
                return createOptimalConfigCommon(config,
                                                 dnnlConvolutionTypeMapping,
                                                 dnnlFCLayoutConfig,
                                                 fcMappingNotation);
            },
            // acceptsShapes
            []([[maybe_unused]] const FCAttrs& attrs,
               const MemoryArgs& memory) -> bool {
                const auto inRank = memory.at(ARG_SRC)->getShape().getRank();
                const auto& inDims = memory.at(ARG_SRC)->getShape().getDims();
                const auto& weightDims = memory.at(ARG_WEI)->getShape().getDims();
                // for original inner product semantics:
                //  when input is 2D tensor -> M in oneDNN will map to widthInConv
                //  when input is 3D tensor -> M in oneDNN will map to widthInConv*minibatch
                // currently nwc mapping in brg::
                //  when input is 2D tensor -> widthInConv will map to 'w', 'n' will be 1
                //  when input is 3D tensor -> widthInConv will map to 'w', 'n' will be minibatch
                Dim widthInConv = inDims[inRank - 2];
                Dim K = inDims[inRank - 1];
                Dim N = weightDims[0];

                const auto& weightsSize = memory.at(ARG_WEI)->getDesc().getCurrentMemSize();
                // Disable Conv1x1 when weight size >= 16M to avoid different weight layout when having different input
                // activation shapes. As a consuquence, peak memory consumption in LLM can be decreased.
                VERIFY(weightsSize < (16 * 1 << 20), " weights size is to big");
                const bool width_in_range = widthInConv >= 2 && widthInConv <= 3136;
                const bool k_in_range = K >= 96 && K <= 4096;
                const bool n_in_range = N >= 96 && N <= K * 4;
                const bool all_conditions_met = width_in_range && k_in_range && n_in_range;
                VERIFY(all_conditions_met, HEURISTICS_MISMATCH);

                return true;
            },
            // create
            [](const FCAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr& context) -> ExecutorPtr {
                struct ConvolutionInstantiator {
                    std::shared_ptr<DnnlConvolutionPrimitive> operator()(
                        const MemoryArgs& memory,
                        const FCAttrs& attrs,
                        const ExecutorContext::CPtr& context,
                        const std::shared_ptr<DnnlShapeAgnosticData>& shareAgnosticData) const {

                        const bool fcSemantic = true;
                        const bool hasBias = !memory.at(ARG_BIAS)->getDesc().empty();
                        ConvAttrs convAttrs{{1}, {0}, {0}, {0},
                                            AutoPaddingType::None, hasBias, attrs.weightsNonTransposed,
                                            false, false, fcSemantic, false, ZeroPointsType::None, {}, attrs.postOps};

                        auto primitive =
                            DefaultInstantiator<DnnlConvolutionPrimitive, ConvAttrs, DnnlShapeAgnosticData>{}(
                            memory,
                            convAttrs,
                            context,
                            shareAgnosticData);

                        // only brgconv_avx512_1x1 primitive is acceptable from the performance perspective
                        if (!primitive || primitive->implType() != brgconv_avx512_1x1) {
                            return nullptr;
                        }

                        return primitive;
                    }
                };

                return std::make_shared<
                    DnnlExecutor<DnnlConvolutionPrimitive, FCAttrs, DnnlShapeAgnosticData, ConvolutionInstantiator>>(
                    attrs,
                    memory,
                    context,
                    false);
            })
        OV_CPU_INSTANCE_ACL(
            "fullyconnected_acl",
            ExecutorType::Acl,
            OperationType::FullyConnected,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                VERIFY(ACLFullyConnectedExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            // createOptimalConfig
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return createOptimalConfigCommon(config,
                                                 aclFCTypeMapping,
                                                 aclFCLayoutConfig,
                                                 fcMappingNotation);
            },
            AcceptsAnyShape<FCAttrs>,
            CreateDefault<ACLFullyConnectedExecutor, FCAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "fullyconnected_acl_lowp",
            ExecutorType::Acl,
            OperationType::FullyConnected,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                VERIFY(ACLLowpFullyConnectedExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            // createOptimalConfig
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return createOptimalConfigCommon(config,
                                                 aclLowpFCTypeMapping,
                                                 aclFCLayoutConfig,
                                                 fcMappingNotation);
            },
            // acceptsShapes
            []([[maybe_unused]] const FCAttrs& attrs,
               const MemoryArgs& memory) -> bool {
                const auto dequantizationScales = getDeQuantizedScales(memory);
                bool isPerChannelQuantization = dequantizationScales.size() > 1;
                // per-channel quantization is not unsupported by ACL
                return !isPerChannelQuantization;
            },
            CreateDefault<ACLLowpFullyConnectedExecutor, FCAttrs>{}
            )
        OV_CPU_INSTANCE_KLEIDIAI(
            "fullyconnected_kleidiai",
            ExecutorType::Kleidiai,
            OperationType::MatMul,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noPostOps(config), UNSUPPORTED_POST_OPS);
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(all_of(f32, srcType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(any_of(weiType(config), f32, i8, i4), UNSUPPORTED_WEI_PRECISIONS);
                VERIFY(implication(hasBias(config), biaType(config) == f32), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(weiRank(config) == 2U, UNSUPPORTED_WEI_RANK);
                VERIFY(MatMulKleidiAIExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            HasNoOptimalConfig<FCAttrs>{},
            AcceptsAnyShape<FCAttrs>,
            CreateDefault<MatMulKleidiAIExecutor, FCAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL(
            "matmul_dnnl",
            ExecutorType::Dnnl,
            OperationType::MatMul,
            // supports
            []([[maybe_unused]] const FCConfig& config) -> bool {
                CPU_DEBUG_CAP_ENABLE(
                    if (getEnvBool("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC")) {
                        VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                        return true;
                    })
                VERIFY(dnnlMatMulSupportedPrecision(config), UNSUPPORTED_SRC_WEI_PRECISIONS);
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(weiRank(config) == 3U, UNSUPPORTED_WEI_RANK);
                VERIFY(weiDims(config)[0] > 1, UNSUPPORTED_WEI_RANK);
                return true;
            },
            // createOptimalConfig
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return createOptimalConfigCommon(config,
                                                 dnnlMatMulTypeMapping,
                                                 dnnlFCLayoutConfig,
                                                 fcMappingNotation);
            },
            AcceptsAnyShape<FCAttrs>,
            // create
            [](const FCAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr& context) -> ExecutorPtr {
                const bool hasBias = !memory.at(ARG_BIAS)->getDesc().empty();
                MatMulAttrs matMulAttrs {
                    false,
                    true,
                    hasBias,
                    attrs.weightsNonTransposed,
                    false,
                    true,
                    true,
                    0,
                    {},
                    attrs.postOps
                };

                return std::make_shared<
                    DnnlExecutor<DnnlMatMulPrimitive, MatMulAttrs, DnnlShapeAgnosticData,
                                 DefaultInstantiator<DnnlMatMulPrimitive, MatMulAttrs, DnnlShapeAgnosticData>>>(
                    matMulAttrs,
                    memory,
                    context,
                    false);
            })
        OV_CPU_INSTANCE_DNNL(
            "fullyconnected_dnnl",
            ExecutorType::Dnnl,
            OperationType::FullyConnected,
            SupportsAnyConfig<FCAttrs>{},
            // createOptimalConfig
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                // Use transformer-optimized config that declares AB8b24a for critical dimensions
                return createTransformerOptimalConfig(config);
            },
            AcceptsAnyShape<FCAttrs>,
            CreateDnnlDefault<DnnlFCPrimitive, FCAttrs>{false, true}
            )
    };

    return fullyconnectedImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
