# test_llama_ops.py

# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import unittest
import torch
from torch._inductor.test_case import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch._C import _xpu_getCurrentRawStream as get_raw_stream

# ------------------- Kernel Imports -------------------
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

    @staticmethod
    def rand_strided(size, stride, dtype, device):
        is_bf16 = dtype == torch.bfloat16
        temp_dtype = torch.float32 if is_bf16 else dtype
        try:
            tensor = torch.empty_strided(
                size, stride, dtype=temp_dtype, device=device
            ).normal_()
        except Exception:
            max_pos = sum((s - 1) * st for s, st in zip(size, stride) if s > 0)
            buffer_size = max_pos + 1
            buffer = torch.randn(buffer_size, device=device, dtype=temp_dtype)
            tensor = torch.as_strided(buffer, size, stride)
        return tensor.to(dtype) if is_bf16 else tensor

    def setUp(self):
        super().setUp()
        device = self.device

        # --- Define Symbolic Sizes and Plain Python types ---
        self.s3 = 131
        self.s16 = 132
        self.s63 = 131
        self.arg4_1 = self.s16
        self.arg6_1 = 131
        self.arg15_1 = 131
        self.arg16_1 = self.s3
        self.arg29_1 = 131
        self.arg31_1 = self.s63
        self.arg41_1 = 1

        # --- Grouped Tensor Initializations to Reduce Redundancy ---
        # For test isolation, we .clone() the template tensor for each argument.

        # Group A: (4, 1), int64
        t_group_a = self.rand_strided((4, 1), (1, 1), device=device, dtype=torch.int64)
        self.arg0_1 = t_group_a.clone()
        self.arg3_1 = t_group_a.clone()

        # Group B: (128256, 2048), bfloat16
        t_group_b = self.rand_strided(
            (128256, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        self.arg1_1 = t_group_b.clone()
        self.arg42_1 = (
            t_group_b.clone()
        )  # Note: This arg was in the original code, re-added for completeness

        # Group C: (4, 2, 131, 128), bfloat16
        # Covers arg7, arg17, arg30, arg32 since s3 and s63 are both 131
        t_group_c = self.rand_strided(
            (4, 2, 131, 128),
            (33536, 16768, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        self.arg7_1 = t_group_c.clone()
        self.arg17_1 = t_group_c.clone()
        self.arg30_1 = t_group_c.clone()
        self.arg32_1 = t_group_c.clone()

        # Group D: CPU scalar, float64
        t_group_d = torch.randn((), device="cpu", dtype=torch.float64)
        self.arg9_1 = t_group_d.clone()
        self.arg10_1 = t_group_d.clone()
        self.arg19_1 = t_group_d.clone()
        self.arg24_1 = t_group_d.clone()
        self.arg34_1 = t_group_d.clone()
        self.arg39_1 = t_group_d.clone()

        # Group E: (2048,), bfloat16
        t_group_e = self.rand_strided(
            (2048,), (1,), device=device, dtype=torch.bfloat16
        )
        self.arg11_1 = t_group_e.clone()
        self.arg20_1 = t_group_e.clone()
        self.arg25_1 = t_group_e.clone()
        self.arg35_1 = t_group_e.clone()
        self.arg40_1 = t_group_e.clone()

        # Group F: (2048, 2048), bfloat16
        t_group_f = self.rand_strided(
            (2048, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        self.arg12_1 = t_group_f.clone()
        self.arg18_1 = t_group_f.clone()
        self.arg26_1 = t_group_f.clone()
        self.arg33_1 = t_group_f.clone()

        # Group G: (256, 2048), bfloat16
        t_group_g = self.rand_strided(
            (256, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        self.arg13_1 = t_group_g.clone()
        self.arg14_1 = t_group_g.clone()
        self.arg27_1 = t_group_g.clone()
        self.arg28_1 = t_group_g.clone()

        # Group H: (128, 2048), bfloat16
        t_group_h = self.rand_strided(
            (128, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        self.arg21_1 = t_group_h.clone()
        self.arg22_1 = t_group_h.clone()
        self.arg36_1 = t_group_h.clone()
        self.arg37_1 = t_group_h.clone()

        # Group I: (2048, 128), bfloat16
        t_group_i = self.rand_strided(
            (2048, 128), (128, 1), device=device, dtype=torch.bfloat16
        )
        self.arg23_1 = t_group_i.clone()
        self.arg38_1 = t_group_i.clone()

        # --- Unique Tensors ---
        # These have unique creation parameters
        self.arg2_1 = self.rand_strided((1,), (1,), device=device, dtype=torch.int64)
        self.arg5_1 = self.rand_strided(
            (4, self.s16), (self.s16, 1), device=device, dtype=torch.int64
        )
        self.arg8_1 = self.rand_strided((64,), (1,), device=device, dtype=torch.float32)

        self.stream0 = get_raw_stream(torch.xpu.current_device())

    def tearDown(self):
        torch.xpu.synchronize()
        super().tearDown()

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3(
        self, device
    ):
        buf2 = self.rand_strided(
            (512, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        buf4 = self.rand_strided(
            (4, 64, 128), (8192, 128, 1), device=device, dtype=torch.float32
        )
        buf8 = torch.empty_strided(
            (4, 16, 128, 128),
            (262144, 16384, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3.run(
            buf2, buf4, self.arg6_1.item(), buf8, 1048576, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4(
        self, device
    ):
        buf6 = self.rand_strided(
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
            self.arg2_1, self.arg4_1, buf11, buf34, 65536, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused_add_cat_mul_2(self, device):
        buf5 = self.rand_strided(
            (512, 256), (256, 1), device=device, dtype=torch.bfloat16
        )
        buf4 = self.rand_strided(
            (4, 64, 128), (8192, 128, 1), device=device, dtype=torch.float32
        )
        buf6 = torch.empty_strided(
            (4, 2, 128, 128), (32768, 128, 256, 1), device=device, dtype=torch.bfloat16
        )
        triton_poi_fused_add_cat_mul_2.run(
            buf5, buf4, self.arg6_1.item(), buf6, 512, 256, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused_clone_6(self, device):
        buf13 = self.rand_strided(
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
        buf1 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )
        triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0.run(
            self.arg0_1,
            self.arg1_1,
            self.arg11_1,
            self.arg10_1.item(),
            buf1,
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)

    def test_triton_poi_fused__to_copy_1(self, device):
        buf3 = torch.empty_strided(
            (4, 1, 1), (1, 4, 4), device=device, dtype=torch.float32
        )
        triton_poi_fused__to_copy_1.run(self.arg3_1, buf3, 4, stream=self.stream0)
        self.assertTrue(True)

    def test_triton_poi_fused_cat_2(self, device):
        buf5 = self.rand_strided(
            (4, 256), (256, 1), device=device, dtype=torch.bfloat16
        )
        buf4 = self.rand_strided(
            (4, 64, 1), (64, 1, 1), device=device, dtype=torch.float32
        )
        buf6 = torch.empty_strided(
            (4, 2, 132, 128),
            (33792, 16896, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        triton_poi_fused_cat_2.run(
            self.arg7_1,
            buf5,
            buf4,
            self.arg9_1.item(),
            buf6,
            135168,
            stream=self.stream0,
        )
        self.assertTrue(True)

    def test_triton_poi_fused_cat_3(self, device):
        s3 = self.s3
        buf7 = self.rand_strided(
            (4, 256), (256, 1), device=device, dtype=torch.bfloat16
        )
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
            self.arg17_1, buf7, buf8, ps0, s3, ps1, xnumel, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4(
        self, device
    ):
        buf2 = self.rand_strided(
            (4, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        buf4 = self.rand_strided(
            (4, 64, 1), (64, 1, 1), device=device, dtype=torch.float32
        )
        buf9 = torch.empty_strided(
            (4, 16, 1, 128), (2048, 128, 128, 1), device=device, dtype=torch.bfloat16
        )
        triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4.run(
            buf2, buf4, self.arg9_1.item(), buf9, 8192, stream=self.stream0
        )
        self.assertTrue(True)

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5(
        self, device
    ):
        buf6 = self.rand_strided(
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
        s3 = self.s3
        ps0 = 1 + s3
        ps1 = 128 + 128 * s3
        ps2 = 2048 + 2048 * s3
        xnumel = 8192 + 8192 * s3
        buf8 = self.rand_strided(
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
        buf18 = self.rand_strided(
            (4, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        buf20 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7.run(
            self.arg0_1,
            self.arg1_1,
            buf18,
            self.arg20_1,
            self.arg19_1.item(),
            buf20,
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)

    def test_triton_poi_fused_mul_silu_8(self, device):
        buf22 = self.rand_strided(
            (4, 128), (128, 1), device=device, dtype=torch.bfloat16
        )
        buf23 = self.rand_strided(
            (4, 1, 128), (128, 128, 1), device=device, dtype=torch.bfloat16
        )
        triton_poi_fused_mul_silu_8.run(buf23, buf22, 512, stream=self.stream0)
        self.assertTrue(True)

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10(
        self, device
    ):
        buf18 = self.rand_strided(
            (4, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        buf24 = self.rand_strided(
            (4, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        buf26 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10.run(
            self.arg0_1,
            self.arg1_1,
            buf18,
            buf24,
            self.arg25_1,
            self.arg24_1.item(),
            buf26,
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9(self, device):
        buf42 = self.rand_strided(
            (4, 1, 2048), (2048, 8192, 1), device=device, dtype=torch.bfloat16
        )
        buf24 = self.rand_strided(
            (4, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        buf41 = self.rand_strided(
            (4, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        buf44 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )
        triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9.run(
            buf42,
            self.arg0_1,
            self.arg1_1,
            buf24,
            buf41,
            self.arg35_1,
            self.arg34_1.item(),
            buf44,
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)

    def test_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11(self, device):
        buf48 = self.rand_strided(
            (4, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        buf50 = self.rand_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(
            buf50,
            buf48,
            self.arg40_1,
            self.arg39_1.item(),
            4,
            2048,
            stream=self.stream0,
        )
        self.assertTrue(True)


# The rest of the file remains the same
instantiate_device_type_tests(
    TestCompiledLlamaOps, globals(), only_for=("xpu"), allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
