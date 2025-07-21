# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
from torch._C import _xpu_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/f7/cf7zzbg2oxnfb4ql25p4mtfmzdx2fobh3zmtqmcyktou7stib2nv.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, rsqrt, hidden_states_1, to_5, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type_4
#   hidden_states_1 => mul_2
#   hidden_states_2 => mul_3
#   inputs_embeds => embedding
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_5 => convert_element_type_5
#   variance => mean
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_4, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %convert_element_type_default_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.float32), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %convert_element_type_default_2), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %rsqrt), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg8_1, %convert_element_type_5), kwargs = {})
triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 2048},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': 'fp64', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 128256")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 2048*tmp4), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp22 = in_ptr3
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp14 = tmp0 + tmp13
        tmp15 = tmp0 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp0)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 128256")
        tmp18 = tl.load(in_ptr1 + (r0_1 + 2048*tmp16), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = 2048.0
        tmp21 = (tmp10 / tmp20)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp19 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp12 * tmp27
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp28, xmask & r0_mask)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/lr/clrzgl33boyhrackjtdqxgjludruvrusepngfiqihsmw4tjsuajk.py
# Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   position_ids_expanded => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_2, torch.float32), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/df/cdf2buhf6nkos5gppynbpy3lqvnhjyd3lsfz5h5a2ylcmg7lonu7.py
# Topologically Sorted Source Nodes: [mul_6, cat_2, mul_7, k_embed], Original ATen: [aten.mul, aten.cat, aten.add]
# Source node to ATen node mapping:
#   cat_2 => cat_1
#   k_embed => add_3
#   mul_6 => mul_6
#   mul_7 => mul_7
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_4), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_6], -1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_5), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
triton_poi_fused_add_cat_mul_2 = async_compile.triton('triton_poi_fused_add_cat_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': 'fp64', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x2 = (xindex % 128)
    y0 = (yindex % 128)
    y1 = yindex // 128
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x5 + 256*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0 + 128*((x2 % 64)) + 8192*y1), ymask & xmask, eviction_policy='evict_last')
    tmp3 = in_ptr2
    tmp2 = tl_math.cos(tmp1)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = x2
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1, 1], 64, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = tl.load(in_ptr0 + (64 + 128*x3 + 256*y4 + (x2)), xmask & ymask & tmp12, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = -tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tmp8 >= tmp11
    tmp18 = tl.full([1, 1], 128, tl.int64)
    tmp19 = tmp8 < tmp18
    tmp20 = tl.load(in_ptr0 + (128*x3 + 256*y4 + ((-64) + x2)), xmask & ymask & tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp12, tmp16, tmp20)
    tmp22 = tl_math.sin(tmp1)
    tmp23 = tmp22 * tmp4
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp21 * tmp24
    tmp26 = tmp7 + tmp25
    tl.store(out_ptr0 + (x5 + 256*y4), tmp26, xmask & ymask)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/qm/cqmvyybj3tctrrlu32oxuz2zhtxzbjx55e67o7fqwxobvok3sarx.py
# Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query, attn_output], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_fused_attention_overrideable, full_default_1, full_default_2, where
#   cat_1 => cat
#   mul_4 => mul_4
#   mul_5 => mul_5
#   q_embed => add_2
#   query => clone_4
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_4), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %_scaled_dot_product_fused_attention_overrideable : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default](args = (%clone_4, %view_18, %view_19, %where), kwargs = {scale: 0.08838834764831845})
triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': 'fp64', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = (xindex % 128)
    x2 = ((xindex // 2048) % 128)
    x3 = xindex // 262144
    x5 = xindex // 128
    x1 = ((xindex // 128) % 16)
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 128*((x0 % 64)) + 8192*x3), None, eviction_policy='evict_last')
    tmp3 = in_ptr2
    tmp2 = tl_math.cos(tmp1)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 * tmp6
    tmp8 = x0
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1], 64, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = tl.load(in_ptr0 + (64 + 128*x5 + (x0)), tmp12, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = -tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tmp8 >= tmp11
    tmp18 = tl.full([1], 128, tl.int64)
    tmp19 = tmp8 < tmp18
    tmp20 = tl.load(in_ptr0 + (128*x5 + ((-64) + x0)), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp12, tmp16, tmp20)
    tmp22 = tl_math.sin(tmp1)
    tmp23 = tmp22 * tmp4
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp21 * tmp24
    tmp26 = tmp7 + tmp25
    tl.store(out_ptr0 + (x0 + 128*x2 + 16384*x1 + 262144*x3), tmp26, None)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/3a/c3axzdjcjzbyzz7uxp6y34qea6lvhdfkly4qsw5t473zbs4lxyfu.py
# Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query, attn_output], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_fused_attention_overrideable, full_default_1, full_default_2, where
#   cat_1 => cat
#   mul_4 => mul_4
#   mul_5 => mul_5
#   q_embed => add_2
#   query => clone_4
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_4), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %_scaled_dot_product_fused_attention_overrideable : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default](args = (%clone_4, %view_18, %view_19, %where), kwargs = {scale: 0.08838834764831845})
triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4 = async_compile.triton('triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 128)
    x2 = ((xindex // 16384) % 16)
    x3 = xindex // 262144
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*(x2 // 8) + 256*x1 + 32768*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/xn/cxnca5l5ieexefqkujzqfuciyenn7jfmoyboie6dg6y6xg6vzybb.py
# Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query, attn_output, mul_13, cat_3, mul_14, q_embed_1, query_1, attn_output_4], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_fused_attention_overrideable, full_default_1, full_default_2, where
#   attn_output_4 => _scaled_dot_product_fused_attention_overrideable_1, full_default_3, full_default_4, where_1
#   cat_1 => cat
#   cat_3 => cat_2
#   mul_13 => mul_14
#   mul_14 => mul_15
#   mul_4 => mul_4
#   mul_5 => mul_5
#   q_embed => add_2
#   q_embed_1 => add_9
#   query => clone_4
#   query_1 => clone_8
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_4), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %_scaled_dot_product_fused_attention_overrideable : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default](args = (%clone_4, %view_18, %view_19, %where), kwargs = {scale: 0.08838834764831845})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_13, %unsqueeze_8), kwargs = {})
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_2, %slice_19], -1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_2, %unsqueeze_9), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %mul_15), kwargs = {})
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_9,), kwargs = {memory_format: torch.contiguous_format})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: xpu:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_4, %full_default_3), kwargs = {})
#   %_scaled_dot_product_fused_attention_overrideable_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default](args = (%clone_8, %view_38, %view_39, %where_1), kwargs = {scale: 0.08838834764831845})
triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5 = async_compile.triton('triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 128) % 128)
    x0 = (xindex % 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp1 <= tmp0
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp6 = (tmp5 != 0)
    tmp7 = tmp4 & tmp6
    tmp8 = 0.0
    tmp9 = float("-inf")
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/we/cwe6d74scfs44fd6j7suxxggjrmxckh6e7mytfetks6aheih263x.py
# Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_1 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 16)
    x2 = ((xindex // 2048) % 128)
    x3 = xindex // 262144
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 16384*x1 + 262144*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/p4/cp4vpwazp4u6ahclzj6oo4oho5qkjs7huglmfpapu4r7s4gdhidt.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_5, hidden_states_6, pow_2, variance_1, rsqrt_1, hidden_states_7, to_7, hidden_states_8], Original ATen: [aten.embedding, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   hidden_states_5 => add_5
#   hidden_states_6 => convert_element_type_14
#   hidden_states_7 => mul_8
#   hidden_states_8 => mul_9
#   inputs_embeds => embedding
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to_7 => convert_element_type_15
#   variance_1 => mean_1
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_22), kwargs = {})
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_14, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %convert_element_type_default_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg14_1, torch.float32), kwargs = {})
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, %convert_element_type_default_3), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %rsqrt_1), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg15_1, %convert_element_type_15), kwargs = {})
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': 'fp64', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 128256")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 2048*tmp4), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp26 = in_ptr4
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp16 = tmp0 + tmp15
        tmp17 = tmp0 < 0
        tmp18 = tl.where(tmp17, tmp16, tmp0)
        tl.device_assert(((0 <= tmp18) & (tmp18 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp18 < 128256")
        tmp20 = tl.load(in_ptr1 + (r0_1 + 2048*tmp18), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 2048.0
        tmp25 = (tmp12 / tmp24)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp23 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp14 * tmp31
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp32, xmask & r0_mask)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/ia/ciaysg47jf5ik7iloigiv2iowmxe4bteop4scwstyr6ozhrfsza3.py
# Topologically Sorted Source Nodes: [silu, mul_10], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul_10 => mul_11
#   silu => convert_element_type_18, convert_element_type_19, mul_10, sigmoid
# Graph fragment:
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_24, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_18,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %sigmoid), kwargs = {})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %view_26), kwargs = {})
triton_poi_fused_mul_silu_8 = async_compile.triton('triton_poi_fused_mul_silu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/sc/cscyouiniggkkumoobqmyo35kimq5tdfo7kmvyfy2gfa6ynetmu4.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_5, hidden_states_9, hidden_states_10, pow_3, variance_2, rsqrt_2, hidden_states_11, to_9, hidden_states_12], Original ATen: [aten.embedding, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   hidden_states_10 => convert_element_type_24
#   hidden_states_11 => mul_12
#   hidden_states_12 => mul_13
#   hidden_states_5 => add_5
#   hidden_states_9 => add_7
#   inputs_embeds => embedding
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   to_9 => convert_element_type_25
#   variance_2 => mean_2
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_22), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_28), kwargs = {})
#   %convert_element_type_24 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.float32), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_24, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %convert_element_type_default_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg19_1, torch.float32), kwargs = {})
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, %convert_element_type_default_4), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %rsqrt_2), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12, torch.bfloat16), kwargs = {})
#   %mul_13 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg20_1, %convert_element_type_25), kwargs = {})
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': 'fp64', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 128256")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 2048*tmp4), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp30 = in_ptr5
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr2 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp18 = tmp0 + tmp17
        tmp19 = tmp0 < 0
        tmp20 = tl.where(tmp19, tmp18, tmp0)
        tl.device_assert(((0 <= tmp20) & (tmp20 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp20 < 128256")
        tmp22 = tl.load(in_ptr1 + (r0_1 + 2048*tmp20), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tmp22 + tmp23
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = 2048.0
        tmp29 = (tmp14 / tmp28)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp27 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp16 * tmp35
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp36, xmask & r0_mask)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/hf/chfwchyjlbue737papatdfcs6lutrnq6wubp7movtpd756u5uqsn.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_5, hidden_states_9, hidden_states_15, hidden_states_16, pow_4, variance_3, rsqrt_3, hidden_states_17, to_11, hidden_states_18], Original ATen: [aten.embedding, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   hidden_states_15 => add_11
#   hidden_states_16 => convert_element_type_34
#   hidden_states_17 => mul_18
#   hidden_states_18 => mul_19
#   hidden_states_5 => add_5
#   hidden_states_9 => add_7
#   inputs_embeds => embedding
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   to_11 => convert_element_type_35
#   variance_3 => mean_3
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_22), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_28), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_42), kwargs = {})
#   %convert_element_type_34 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float32), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_34, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %convert_element_type_default_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg25_1, torch.float32), kwargs = {})
#   %add_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, %convert_element_type_default_5), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_34, %rsqrt_3), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18, torch.bfloat16), kwargs = {})
#   %mul_19 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg26_1, %convert_element_type_35), kwargs = {})
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': 'fp64', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 128256")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 2048*tmp4), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp10 = tmp8 + tmp9
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
        tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp12, xmask & r0_mask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp23 = in_ptr5
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp18 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = 2048.0
        tmp22 = (tmp16 / tmp21)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp20 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp18 * tmp28
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp29, xmask & r0_mask)
''', device_str='xpu')


# kernel path: /workspace1/xingyuan/20250530-llama31-profiling/torchinductor_cache/a2/ca2xt6lwmvlnrnmlgqh5hnjjhcwnxq5izuvo77twxvr56pxcvk7q.py
# Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, pow_5, variance_4, rsqrt_4, hidden_states_21, to_13, hidden_states_22], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   hidden_states_19 => add_13
#   hidden_states_20 => convert_element_type_44
#   hidden_states_21 => mul_22
#   hidden_states_22 => mul_23
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
#   to_13 => convert_element_type_45
#   variance_4 => mean_4
# Graph fragment:
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %convert_element_type_44 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.float32), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_44, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %convert_element_type_default_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg30_1, torch.float32), kwargs = {})
#   %add_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, %convert_element_type_default_6), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_4,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_44, %rsqrt_4), kwargs = {})
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_22, torch.bfloat16), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg31_1, %convert_element_type_45), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': 'fp64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'driver_version': '1.6.33276+22', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '8626E6AF5A7AFAC5EC8787A795AE611A7201C920A071BB715D00D43736F8E36D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp15 = in_ptr2
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_out_ptr0 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr0 + (r0_1 + 2048*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = 2048.0
        tmp14 = (tmp6 / tmp13)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp12 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp8 * tmp20
        tl.store(in_out_ptr0 + (r0_1 + 2048*x0), tmp21, xmask & r0_mask)
''', device_str='xpu')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1 = args
    args.clear()
    s22 = arg12_1
    s35 = arg32_1
    assert_size_stride(arg0_1, (4, 128), (128, 1))
    assert_size_stride(arg1_1, (128256, 2048), (2048, 1))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (4, 128), (128, 1))
    assert_size_stride(arg4_1, (4, 128), (0, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (), ())
    assert_size_stride(arg7_1, (), ())
    assert_size_stride(arg8_1, (2048, ), (1, ))
    assert_size_stride(arg9_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg10_1, (256, 2048), (2048, 1))
    assert_size_stride(arg11_1, (256, 2048), (2048, 1))
    assert_size_stride(arg13_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg14_1, (), ())
    assert_size_stride(arg15_1, (2048, ), (1, ))
    assert_size_stride(arg16_1, (128, 2048), (2048, 1))
    assert_size_stride(arg17_1, (128, 2048), (2048, 1))
    assert_size_stride(arg18_1, (2048, 128), (128, 1))
    assert_size_stride(arg19_1, (), ())
    assert_size_stride(arg20_1, (2048, ), (1, ))
    assert_size_stride(arg21_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg22_1, (256, 2048), (2048, 1))
    assert_size_stride(arg23_1, (256, 2048), (2048, 1))
    assert_size_stride(arg24_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg25_1, (), ())
    assert_size_stride(arg26_1, (2048, ), (1, ))
    assert_size_stride(arg27_1, (128, 2048), (2048, 1))
    assert_size_stride(arg28_1, (128, 2048), (2048, 1))
    assert_size_stride(arg29_1, (2048, 128), (128, 1))
    assert_size_stride(arg30_1, (), ())
    assert_size_stride(arg31_1, (2048, ), (1, ))
    assert_size_stride(arg33_1, (128256, 2048), (2048, 1))
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        buf1 = empty_strided_xpu((4, 128, 2048), (262144, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, rsqrt, hidden_states_1, to_5, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg8_1, arg7_1.item(), buf1, 512, 2048, stream=stream0)
        del arg7_1
        del arg8_1
        buf2 = empty_strided_xpu((512, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg9_1, (2048, 2048), (1, 2048), 0), out=buf2)
        del arg9_1
        buf3 = empty_strided_xpu((4, 1, 128), (128, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg3_1, buf3, 512, stream=stream0)
        del arg3_1
        buf4 = empty_strided_xpu((4, 64, 128), (8192, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [position_ids_expanded, matmul], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg5_1, (4, 64, 1), (0, 1, 0), 0), buf3, out=buf4)
        del arg5_1
        del buf3
        buf5 = empty_strided_xpu((512, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg10_1, (2048, 256), (1, 2048), 0), out=buf5)
        del arg10_1
        buf6 = empty_strided_xpu((4, 2, 128, 128), (32768, 128, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul_6, cat_2, mul_7, k_embed], Original ATen: [aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_mul_2.run(buf5, buf4, arg6_1.item(), buf6, 512, 256, stream=stream0)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg11_1, (2048, 256), (1, 2048), 0), out=buf7)
        del arg11_1
        buf8 = reinterpret_tensor(buf1, (4, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query, attn_output], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3.run(buf2, buf4, arg6_1.item(), buf8, 1048576, stream=stream0)
        buf9 = reinterpret_tensor(buf2, (4, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query, attn_output], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4.run(buf6, buf9, 1048576, stream=stream0)
        buf10 = empty_strided_xpu((4, 16, 128, 128), (262144, 16384, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query, attn_output], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4.run(buf7, buf10, 1048576, stream=stream0)
        buf11 = empty_strided_xpu((4, 1, 128, 128), (16384, 16384, 128, 1), torch.bfloat16)
        buf34 = empty_strided_xpu((4, 1, 128, 128), (16384, 16384, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query, attn_output, mul_13, cat_3, mul_14, q_embed_1, query_1, attn_output_4], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5.run(arg2_1, arg4_1, buf11, buf34, 65536, stream=stream0)
        del arg2_1
        del arg4_1
        # Topologically Sorted Source Nodes: [mul_4, cat_1, mul_5, q_embed, query, attn_output], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        buf12 = torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default(buf8, buf9, buf10, buf11, scale=0.08838834764831845)
        buf13 = buf12[0]
        assert_size_stride(buf13, (4, 16, 128, 128), (262144, 16384, 128, 1))
        assert_alignment(buf13, 16)
        del buf12
        buf17 = reinterpret_tensor(buf9, (4, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf13, buf17, 1048576, stream=stream0)
        buf18 = reinterpret_tensor(buf13, (512, 2048), (2048, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg13_1, (2048, 2048), (1, 2048), 0), out=buf18)
        del arg13_1
        buf20 = reinterpret_tensor(buf17, (4, 128, 2048), (262144, 2048, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_5, hidden_states_6, pow_2, variance_1, rsqrt_1, hidden_states_7, to_7, hidden_states_8], Original ATen: [aten.embedding, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7.run(arg0_1, arg1_1, buf18, arg15_1, arg14_1.item(), buf20, 512, 2048, stream=stream0)
        del arg14_1
        del arg15_1
        buf21 = reinterpret_tensor(buf11, (512, 128), (128, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg16_1, (2048, 128), (1, 2048), 0), out=buf21)
        del arg16_1
        buf22 = empty_strided_xpu((512, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg17_1, (2048, 128), (1, 2048), 0), out=buf22)
        del arg17_1
        buf23 = reinterpret_tensor(buf21, (4, 128, 128), (16384, 128, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [silu, mul_10], Original ATen: [aten.silu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_8.run(buf23, buf22, 65536, stream=stream0)
        del buf22
        buf24 = reinterpret_tensor(buf20, (512, 2048), (2048, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [down_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 128), (128, 1), 0), reinterpret_tensor(arg18_1, (128, 2048), (1, 128), 0), out=buf24)
        del arg18_1
        buf26 = reinterpret_tensor(buf8, (4, 128, 2048), (262144, 2048, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_5, hidden_states_9, hidden_states_10, pow_3, variance_2, rsqrt_2, hidden_states_11, to_9, hidden_states_12], Original ATen: [aten.embedding, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9.run(arg0_1, arg1_1, buf18, buf24, arg20_1, arg19_1.item(), buf26, 512, 2048, stream=stream0)
        del arg19_1
        del arg20_1
        buf27 = reinterpret_tensor(buf10, (512, 2048), (2048, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg21_1, (2048, 2048), (1, 2048), 0), out=buf27)
        del arg21_1
        buf28 = empty_strided_xpu((512, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg22_1, (2048, 256), (1, 2048), 0), out=buf28)
        del arg22_1
        buf29 = empty_strided_xpu((4, 2, 128, 128), (32768, 128, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul_15, cat_4, mul_16, k_embed_1], Original ATen: [aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_mul_2.run(buf28, buf4, arg6_1.item(), buf29, 512, 256, stream=stream0)
        buf30 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg23_1, (2048, 256), (1, 2048), 0), out=buf30)
        del arg23_1
        buf31 = reinterpret_tensor(buf26, (4, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [mul_13, cat_3, mul_14, q_embed_1, query_1, attn_output_4], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3.run(buf27, buf4, arg6_1.item(), buf31, 1048576, stream=stream0)
        del arg6_1
        del buf4
        buf32 = reinterpret_tensor(buf27, (4, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [mul_13, cat_3, mul_14, q_embed_1, query_1, attn_output_4], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4.run(buf29, buf32, 1048576, stream=stream0)
        buf33 = empty_strided_xpu((4, 16, 128, 128), (262144, 16384, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul_13, cat_3, mul_14, q_embed_1, query_1, attn_output_4], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4.run(buf30, buf33, 1048576, stream=stream0)
        # Topologically Sorted Source Nodes: [mul_13, cat_3, mul_14, q_embed_1, query_1, attn_output_4], Original ATen: [aten.mul, aten.cat, aten.add, aten.clone, aten.scalar_tensor, aten.where, aten._scaled_dot_product_fused_attention_overrideable]
        buf35 = torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default(buf31, buf32, buf33, buf34, scale=0.08838834764831845)
        del buf31
        del buf32
        buf36 = buf35[0]
        assert_size_stride(buf36, (4, 16, 128, 128), (262144, 16384, 128, 1))
        assert_alignment(buf36, 16)
        del buf35
        buf40 = reinterpret_tensor(buf33, (4, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [attn_output_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf36, buf40, 1048576, stream=stream0)
        buf41 = reinterpret_tensor(buf36, (512, 2048), (2048, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [attn_output_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg24_1, (2048, 2048), (1, 2048), 0), out=buf41)
        del arg24_1
        buf42 = reinterpret_tensor(buf18, (4, 128, 2048), (262144, 2048, 1), 0); del buf18  # reuse
        buf44 = reinterpret_tensor(buf40, (4, 128, 2048), (262144, 2048, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states_5, hidden_states_9, hidden_states_15, hidden_states_16, pow_4, variance_3, rsqrt_3, hidden_states_17, to_11, hidden_states_18], Original ATen: [aten.embedding, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10.run(buf42, arg0_1, arg1_1, buf24, buf41, arg26_1, arg25_1.item(), buf44, 512, 2048, stream=stream0)
        del arg0_1
        del arg1_1
        del arg25_1
        del arg26_1
        del buf24
        del buf41
        buf45 = reinterpret_tensor(buf34, (512, 128), (128, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg27_1, (2048, 128), (1, 2048), 0), out=buf45)
        del arg27_1
        buf46 = reinterpret_tensor(buf23, (512, 128), (128, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg28_1, (2048, 128), (1, 2048), 0), out=buf46)
        del arg28_1
        buf47 = reinterpret_tensor(buf45, (4, 128, 128), (16384, 128, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [silu_1, mul_19], Original ATen: [aten.silu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_8.run(buf47, buf46, 65536, stream=stream0)
        del buf46
        buf48 = reinterpret_tensor(buf44, (512, 2048), (2048, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [down_proj_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (512, 128), (128, 1), 0), reinterpret_tensor(arg29_1, (128, 2048), (1, 128), 0), out=buf48)
        del arg29_1
        del buf47
        buf50 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, pow_5, variance_4, rsqrt_4, hidden_states_21, to_13, hidden_states_22], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(buf50, buf48, arg31_1, arg30_1.item(), 512, 2048, stream=stream0)
        del arg30_1
        del arg31_1
        del buf48
        buf51 = empty_strided_xpu((4, 128256), (128256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (4, 2048), (262144, 1), 260096), reinterpret_tensor(arg33_1, (2048, 128256), (1, 2048), 0), out=buf51)
        del arg33_1
        del buf50
    return (128 + s22, buf6, buf29, reinterpret_tensor(buf7, (4, 2, 128, 128), (32768, 128, 256, 1), 0), reinterpret_tensor(buf30, (4, 2, 128, 128), (32768, 128, 256, 1), 0), reinterpret_tensor(buf51, (4, 1, 128256), (128256, 128256, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 128), (128, 1), device='xpu:0', dtype=torch.int64)
    arg1_1 = rand_strided((128256, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((128, ), (1, ), device='xpu:0', dtype=torch.int64)
    arg3_1 = rand_strided((4, 128), (128, 1), device='xpu:0', dtype=torch.int64)
    arg4_1 = rand_strided((4, 128), (0, 1), device='xpu:0', dtype=torch.int64)
    arg5_1 = rand_strided((64, ), (1, ), device='xpu:0', dtype=torch.float32)
    arg6_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg7_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg8_1 = rand_strided((2048, ), (1, ), device='xpu:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((2048, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((256, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((256, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg12_1 = 0
    arg13_1 = rand_strided((2048, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg15_1 = rand_strided((2048, ), (1, ), device='xpu:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((128, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((128, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((2048, 128), (128, 1), device='xpu:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg20_1 = rand_strided((2048, ), (1, ), device='xpu:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((2048, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((256, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((256, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((2048, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg26_1 = rand_strided((2048, ), (1, ), device='xpu:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((128, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((128, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((2048, 128), (128, 1), device='xpu:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg31_1 = rand_strided((2048, ), (1, ), device='xpu:0', dtype=torch.bfloat16)
    arg32_1 = 1
    arg33_1 = rand_strided((128256, 2048), (2048, 1), device='xpu:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
