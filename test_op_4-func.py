from op_4_func import (
    triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3,
    triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4,
    triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5,
    triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4,
    triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5,
    triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_6,
    triton_poi_fused__to_copy_1,
    triton_poi_fused_add_cat_mul_2,
    triton_poi_fused_cat_2,
    triton_poi_fused_cat_3,
    triton_poi_fused_clone_6,
    triton_poi_fused_mul_silu_8,
    triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10,
    triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7,
    triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9,
    triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11,
    triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0,
)

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


# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import dataclasses
import functools
import io
import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from importlib.machinery import SourceFileLoader
from pathlib import Path
from string import Template
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd, config
from torch._dynamo.backends.debugging import aot_eager
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.overrides import BaseTorchFunctionMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_S390X,
    parametrize,
    scoped_load_inline,
    skipIfWindows,
)
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_CUDA, HAS_GPU
from torch.testing._internal.logging_utils import logs_to_string
from torch.utils._python_dispatch import TorchDispatchMode


class TestCompiledLlamaOps(TestCase):
    def setUp(self) -> None:
        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(config.patch("record_runtime_overhead", False))
        super().setUp()

    def tearDown(self) -> None:
        self.exit_stack.close()
        super().tearDown()

    def check_output_and_recompiles(
        self, device, shapes
    ):
        if isinstance(count, list):
            captures, compiles = count
        else:
            captures, compiles = count, count
        with torch.autograd.set_multithreading_enabled(False):
            torch._dynamo.reset()
            counters["compiled_autograd"].clear()
            torch.manual_seed(123)
            expected = list(fn())
            torch.manual_seed(123)
            with (
                compiled_autograd._enable(compiler_fn),
                mock.patch(
                    "torch._functorch.aot_autograd.AOT_COUNTER",
                    new_callable=itertools.count,
                ),
            ):
                opt_fn = torch.compile(fn) if compile_fn else fn
                actual = list(opt_fn())
            self.assertEqual(expected, actual)
            self.assertEqual(counters["compiled_autograd"]["captures"], captures)
            self.assertEqual(counters["compiled_autograd"]["compiles"], compiles)

    @parametrize(
        "input_shapes", [[[512, 2048], [4, 64, 128], [], [4, 16, 128, 128], [], []]]
    )
    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3(
        self, device, input_shapes
    ):
        pass


instantiate_device_type_tests(
    TestLearnableBiases, globals(), only_for=test_device, allow_xpu=True
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
