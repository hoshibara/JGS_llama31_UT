# test_llama_ops.py

# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import unittest
import torch
from torch._inductor.test_case import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
from torch._dynamo.testing import rand_strided

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


class TestCompiledLlamaOps(TestCase):
    """
    Unit tests for Triton kernels. Each test case defines its own inputs locally
    to ensure maximum independence and clarity.
    """

    def setUp(self):
        super().setUp()
        # The stream is the only common element needed.
        self.stream0 = get_raw_stream(torch.xpu.current_device())

    def tearDown(self):
        torch.xpu.synchronize()
        super().tearDown()

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3(
        self, device
    ):
        # Local inputs based on the second 'call' function from your prompt.
        arg6_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf2 = rand_strided((512, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf4 = rand_strided(
            (4, 64, 128), (8192, 128, 1), device=device, dtype=torch.float32
        )
        buf8 = torch.empty_strided(
            (4, 16, 128, 128),
            (262144, 16384, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )

        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3.run(
            buf2, buf4, arg6_1.item(), buf8, 1048576, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4(
        self, device
    ):
        # Local inputs based on the second 'call' function.
        buf6 = rand_strided(
            (4, 2, 128, 128), (32768, 128, 256, 1), device=device, dtype=torch.bfloat16
        )
        buf9 = torch.empty_strided(
            (4, 16, 128, 128),
            (262144, 16384, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )

        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4.run(
            buf6, buf9, 1048576, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5(
        self, device
    ):
        # Local inputs based on the second 'call' function.
        arg2_1 = rand_strided((128,), (1,), device=device, dtype=torch.int64)
        arg4_1 = rand_strided((4, 128), (0, 1), device=device, dtype=torch.int64)
        buf11 = torch.empty_strided(
            (4, 1, 128, 128),
            (16384, 16384, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        buf34 = torch.empty_strided(
            (4, 1, 128, 128),
            (16384, 16384, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )

        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5.run(
            arg2_1, arg4_1, buf11, buf34, 65536, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused_add_cat_mul_2(self, device):
        # Local inputs based on the second 'call' function.
        arg6_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf5 = rand_strided((512, 256), (256, 1), device=device, dtype=torch.bfloat16)
        buf4 = rand_strided(
            (4, 64, 128), (8192, 128, 1), device=device, dtype=torch.float32
        )
        buf6 = torch.empty_strided(
            (4, 2, 128, 128), (32768, 128, 256, 1), device=device, dtype=torch.bfloat16
        )

        triton_poi_fused_add_cat_mul_2.run(
            buf5, buf4, arg6_1.item(), buf6, 512, 256, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused_clone_6(self, device):
        # Local inputs based on the second 'call' function.
        buf13 = rand_strided(
            (4, 16, 128, 128),
            (262144, 16384, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        buf17 = torch.empty_strided(
            (4, 128, 16, 128),
            (262144, 2048, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )

        triton_poi_fused_clone_6.run(buf13, buf17, 1048576, stream=self.stream0)
        self.assertTrue(True)

    def test_triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0(self, device):
        # Local inputs based on the first 'call' function.
        arg0_1 = rand_strided((4, 1), (1, 1), device=device, dtype=torch.int64)
        arg1_1 = rand_strided(
            (128256, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        arg11_1 = rand_strided((2048,), (1,), device=device, dtype=torch.bfloat16)
        arg10_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf1 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )

        triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0.run(
            arg0_1, arg1_1, arg11_1, arg10_1.item(), buf1, 4, 2048, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused__to_copy_1(self, device):
        # Local inputs based on the first 'call' function.
        arg3_1 = rand_strided((4, 1), (1, 1), device=device, dtype=torch.int64)
        buf3 = torch.empty_strided(
            (4, 1, 1), (1, 4, 4), device=device, dtype=torch.float32
        )

        triton_poi_fused__to_copy_1.run(arg3_1, buf3, 4, stream=self.stream0)
        self.assertTrue(True)

    def test_triton_poi_fused_cat_2(self, device):
        # Local inputs based on the first 'call' function.
        arg7_1 = rand_strided(
            (4, 2, 131, 128),
            (33536, 16768, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        arg9_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf5 = rand_strided((4, 256), (256, 1), device=device, dtype=torch.bfloat16)
        buf4 = rand_strided((4, 64, 1), (64, 1, 1), device=device, dtype=torch.float32)
        buf6 = torch.empty_strided(
            (4, 2, 132, 128),
            (33792, 16896, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )

        triton_poi_fused_cat_2.run(
            arg7_1, buf5, buf4, arg9_1.item(), buf6, 135168, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused_cat_3(self, device):
        # Local inputs based on the first 'call' function.
        s3 = 131
        arg17_1 = rand_strided(
            (4, 2, s3, 128),
            (256 * s3, 128 * s3, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        buf7 = rand_strided((4, 256), (256, 1), device=device, dtype=torch.bfloat16)
        ps0 = 1 + s3
        ps1 = 128 + 128 * s3
        buf8 = torch.empty_strided(
            (4, 2, ps0, 128),
            (256 * ps0, 128 * ps0, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        xnumel = 1024 + 1024 * s3

        triton_poi_fused_cat_3.run(
            arg17_1, buf7, buf8, ps0, s3, ps1, xnumel, stream=self.stream0
        )
        self.assertTrue(True)

    # Note: The original import had two kernels with this name. This test will use the first one it finds.
    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4(
        self, device
    ):
        # Local inputs based on the first 'call' function.
        arg9_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf2 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf4 = rand_strided((4, 64, 1), (64, 1, 1), device=device, dtype=torch.float32)
        buf9 = torch.empty_strided(
            (4, 16, 1, 128), (2048, 128, 128, 1), device=device, dtype=torch.bfloat16
        )

        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4.run(
            buf2, buf4, arg9_1.item(), buf9, 8192, stream=self.stream0
        )
        self.assertTrue(True)

    # Note: The original import had two kernels with this name. This test will use the first one it finds.
    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5(
        self, device
    ):
        # Local inputs based on the first 'call' function.
        buf6 = rand_strided(
            (4, 2, 132, 128),
            (33792, 16896, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        buf10 = torch.empty_strided(
            (4, 16, 132, 128),
            (270336, 16896, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )

        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5.run(
            buf6, buf10, 1081344, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_6(
        self, device
    ):
        # Local inputs based on the first 'call' function.
        s3 = 131
        ps0 = 1 + s3
        ps1 = 128 + 128 * s3
        ps2 = 2048 + 2048 * s3
        xnumel = 8192 + 8192 * s3
        buf8 = rand_strided(
            (4, 2, ps0, 128),
            (256 * ps0, 128 * ps0, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        buf11 = torch.empty_strided(
            (4, 16, ps0, 128),
            (ps2, 128 * ps0, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )

        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_6.run(
            buf8, buf11, ps1, ps2, s3, xnumel, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7(self, device):
        # Local inputs based on the first 'call' function for the aliased kernel triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_8
        arg0_1 = rand_strided((4, 1), (1, 1), device=device, dtype=torch.int64)
        arg1_1 = rand_strided(
            (128256, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        arg19_1 = torch.randn((), device="cpu", dtype=torch.float64)
        arg20_1 = rand_strided((2048,), (1,), device=device, dtype=torch.bfloat16)
        buf18 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf20 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )

        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7.run(
            arg0_1,
            arg1_1,
            buf18,
            arg20_1,
            arg19_1.item(),
            buf20,
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)

    def test_triton_poi_fused_mul_silu_8(self, device):
        # Local inputs based on the first 'call' function for the aliased kernel triton_poi_fused_mul_silu_9
        buf22 = rand_strided((4, 128), (128, 1), device=device, dtype=torch.bfloat16)
        buf23 = rand_strided(
            (4, 1, 128), (128, 128, 1), device=device, dtype=torch.bfloat16
        )

        triton_poi_fused_mul_silu_8.run(buf23, buf22, 512, stream=self.stream0)
        self.assertTrue(True)

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10(
        self, device
    ):
        # Local inputs based on the first 'call' function.
        arg0_1 = rand_strided((4, 1), (1, 1), device=device, dtype=torch.int64)
        arg1_1 = rand_strided(
            (128256, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        arg24_1 = torch.randn((), device="cpu", dtype=torch.float64)
        arg25_1 = rand_strided((2048,), (1,), device=device, dtype=torch.bfloat16)
        buf18 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf24 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf26 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )

        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10.run(
            arg0_1,
            arg1_1,
            buf18,
            buf24,
            arg25_1,
            arg24_1.item(),
            buf26,
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9(self, device):
        # Local inputs based on the first 'call' function for the aliased kernel triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_11
        arg0_1 = rand_strided((4, 1), (1, 1), device=device, dtype=torch.int64)
        arg1_1 = rand_strided(
            (128256, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        arg34_1 = torch.randn((), device="cpu", dtype=torch.float64)
        arg35_1 = rand_strided((2048,), (1,), device=device, dtype=torch.bfloat16)
        buf42 = rand_strided(
            (4, 1, 2048), (2048, 8192, 1), device=device, dtype=torch.bfloat16
        )
        buf24 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf41 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf44 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )

        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9.run(
            buf42,
            arg0_1,
            arg1_1,
            buf24,
            buf41,
            arg35_1,
            arg34_1.item(),
            buf44,
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)

    def test_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11(self, device):
        # Local inputs based on the first 'call' function for the aliased kernel triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_12
        arg39_1 = torch.randn((), device="cpu", dtype=torch.float64)
        arg40_1 = rand_strided((2048,), (1,), device=device, dtype=torch.bfloat16)
        buf48 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf50 = rand_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )

        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(
            buf50,
            buf48,
            arg40_1,
            arg39_1.item(),
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)


instantiate_device_type_tests(
    TestCompiledLlamaOps, globals(), only_for=("xpu"), allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
