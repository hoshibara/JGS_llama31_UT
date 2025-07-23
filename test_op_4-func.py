# test_llama_ops.py

# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import unittest
import os
import pathlib
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

    This test suite supports two modes via the 'UT_MODE' environment variable:
    1. DUMP (default): Generates random inputs, runs the kernel, and saves both
       inputs and outputs to the './test_data' directory.
    2. COMPARE: Loads inputs from './test_data', runs the kernel, and compares
       the results against the saved golden outputs.
    """

    # Get UT mode from environment variable, default to DUMP
    UT_MODE = os.environ.get("UT_MODE", "DUMP").upper()
    DATA_DIR = pathlib.Path(__file__).parent / "test_data"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create the data directory if it doesn't exist
        if cls.UT_MODE == "DUMP":
            cls.DATA_DIR.mkdir(exist_ok=True)
            print(f"Running in DUMP mode. Data will be saved to '{cls.DATA_DIR}'.")
        elif cls.UT_MODE == "COMPARE":
            if not cls.DATA_DIR.exists():
                raise FileNotFoundError(
                    f"Running in COMPARE mode, but data directory '{cls.DATA_DIR}' not found. "
                    "Please run in DUMP mode first."
                )
            print(
                f"Running in COMPARE mode. Data will be loaded from '{cls.DATA_DIR}'."
            )
        else:
            raise ValueError(f"Invalid UT_MODE: {cls.UT_MODE}. Choose DUMP or COMPARE.")

    def setUp(self):
        super().setUp()
        # The stream is the only common element needed.
        self.stream0 = get_raw_stream(torch.xpu.current_device())

    def tearDown(self):
        torch.xpu.synchronize()
        super().tearDown()

    def _run_test(
        self, test_name, func, kernel_args, output_indices, in_out_indices, **kwargs
    ):
        """Helper function to handle DUMP and COMPARE logic."""
        input_file = self.DATA_DIR / f"{test_name}_input.pt"
        output_file = self.DATA_DIR / f"{test_name}_output.pt"

        if self.UT_MODE == "DUMP":
            print(f"[DUMP] Running '{test_name}' and saving data...")

            # Save all initial arguments
            # Note: For in_out_ptr, this saves the initial state.
            torch.save({"args": kernel_args, "kwargs": kwargs}, input_file)

            # Run the kernel
            func(*kernel_args, **kwargs)
            torch.xpu.synchronize()

            # Collect output tensors
            outputs_to_save = {
                "outputs": [kernel_args[i].clone() for i in output_indices],
                "in_outs": [kernel_args[i].clone() for i in in_out_indices],
            }
            torch.save(outputs_to_save, output_file)
            self.assertTrue(True)

        else:  # COMPARE mode
            print(f"[COMPARE] Running '{test_name}' and verifying results...")
            if not input_file.exists() or not output_file.exists():
                self.fail(
                    f"Data files for '{test_name}' not found. Run in DUMP mode first."
                )

            # Load inputs and golden outputs
            loaded_data = torch.load(input_file, map_location=self.device)
            run_args = loaded_data["args"]
            run_kwargs = loaded_data["kwargs"]

            golden_outputs = torch.load(output_file, map_location=self.device)
            golden_out_tensors = golden_outputs["outputs"]
            golden_in_out_tensors = golden_outputs["in_outs"]

            # For pure output pointers, we need to create empty tensors with the correct
            # shape/dtype for the kernel to write into.
            for i, idx in enumerate(output_indices):
                run_args[idx] = torch.empty_like(golden_out_tensors[i])

            # Run the kernel with loaded inputs
            func(*run_args, **run_kwargs)
            torch.xpu.synchronize()

            # Compare results
            for i, idx in enumerate(output_indices):
                torch.testing.assert_close(
                    run_args[idx],
                    golden_out_tensors[i],
                    msg=f"Output mismatch at index {idx}",
                )

            for i, idx in enumerate(in_out_indices):
                torch.testing.assert_close(
                    run_args[idx],
                    golden_in_out_tensors[i],
                    msg=f"In-out mismatch at index {idx}",
                )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3(
        self, device
    ):
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
        # def triton_poi_fused__..._where_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK)
        kernel_args = [buf2, buf4, arg6_1.item(), buf8, 1048576]
        self._run_test(
            "test_triton_poi_fused__..._where_3",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3.run,
            kernel_args,
            output_indices=[3],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4(
        self, device
    ):
        buf6 = rand_strided(
            (4, 2, 128, 128), (32768, 128, 256, 1), device=device, dtype=torch.bfloat16
        )
        buf9 = torch.empty_strided(
            (4, 16, 128, 128),
            (262144, 16384, 128, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        # def triton_poi_fused__..._where_4(in_ptr0, out_ptr0, xnumel, XBLOCK)
        kernel_args = [buf6, buf9, 1048576]
        self._run_test(
            "test_triton_poi_fused__..._where_4",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5(
        self, device
    ):
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
        # def triton_poi_fused__..._where_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK)
        kernel_args = [arg2_1, arg4_1, buf11, buf34, 65536]
        self._run_test(
            "test_triton_poi_fused__..._where_5",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5.run,
            kernel_args,
            output_indices=[2, 3],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_add_cat_mul_2(self, device):
        arg6_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf5 = rand_strided((512, 256), (256, 1), device=device, dtype=torch.bfloat16)
        buf4 = rand_strided(
            (4, 64, 128), (8192, 128, 1), device=device, dtype=torch.float32
        )
        buf6 = torch.empty_strided(
            (4, 2, 128, 128), (32768, 128, 256, 1), device=device, dtype=torch.bfloat16
        )
        # def triton_poi_fused_add_cat_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK, XBLOCK)
        kernel_args = [buf5, buf4, arg6_1.item(), buf6, 512, 256]
        self._run_test(
            "test_triton_poi_fused_add_cat_mul_2",
            triton_poi_fused_add_cat_mul_2.run,
            kernel_args,
            output_indices=[3],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_clone_6(self, device):
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
        # def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK)
        kernel_args = [buf13, buf17, 1048576]
        self._run_test(
            "test_triton_poi_fused_clone_6",
            triton_poi_fused_clone_6.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0(self, device):
        arg0_1 = rand_strided((4, 1), (1, 1), device=device, dtype=torch.int64)
        arg1_1 = rand_strided(
            (128256, 2048), (2048, 1), device=device, dtype=torch.bfloat16
        )
        arg11_1 = rand_strided((2048,), (1,), device=device, dtype=torch.bfloat16)
        arg10_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf1 = torch.empty_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )
        # def triton_red_fused__..._rsqrt_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK, R0_BLOCK)
        kernel_args = [arg0_1, arg1_1, arg11_1, arg10_1.item(), buf1, 4, 2048]
        self._run_test(
            "test_triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0",
            triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0.run,
            kernel_args,
            output_indices=[4],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__to_copy_1(self, device):
        arg3_1 = rand_strided((4, 1), (1, 1), device=device, dtype=torch.int64)
        buf3 = torch.empty_strided(
            (4, 1, 1), (1, 4, 4), device=device, dtype=torch.float32
        )
        # def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK)
        kernel_args = [arg3_1, buf3, 4]
        self._run_test(
            "test_triton_poi_fused__to_copy_1",
            triton_poi_fused__to_copy_1.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_cat_2(self, device):
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
        # def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK)
        kernel_args = [arg7_1, buf5, buf4, arg9_1.item(), buf6, 135168]
        self._run_test(
            "test_triton_poi_fused_cat_2",
            triton_poi_fused_cat_2.run,
            kernel_args,
            output_indices=[4],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_cat_3(self, device):
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
        # def triton_poi_fused_cat_3(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK)
        kernel_args = [arg17_1, buf7, buf8, ps0, s3, ps1, xnumel]
        self._run_test(
            "test_triton_poi_fused_cat_3",
            triton_poi_fused_cat_3.run,
            kernel_args,
            output_indices=[2],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4_v2(
        self, device
    ):
        arg9_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf2 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf4 = rand_strided((4, 64, 1), (64, 1, 1), device=device, dtype=torch.float32)
        buf9 = torch.empty_strided(
            (4, 16, 1, 128), (2048, 128, 128, 1), device=device, dtype=torch.bfloat16
        )
        # def triton_poi_fused__..._where_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK)
        kernel_args = [buf2, buf4, arg9_1.item(), buf9, 8192]
        self._run_test(
            "test_triton_poi_fused__..._where_4_v2",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4.run,
            kernel_args,
            output_indices=[3],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5_v2(
        self, device
    ):
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
        # def triton_poi_fused__..._where_5(in_ptr0, out_ptr0, xnumel, XBLOCK)
        kernel_args = [buf6, buf10, 1081344]
        self._run_test(
            "test_triton_poi_fused__..._where_5_v2",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_6(
        self, device
    ):
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
        # def triton_poi_fused__..._where_6(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK)
        kernel_args = [buf8, buf11, ps1, ps2, s3, xnumel]
        self._run_test(
            "test_triton_poi_fused__..._where_6",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_6.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7(self, device):
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
        # def triton_red_fused__..._rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK, R0_BLOCK)
        kernel_args = [arg0_1, arg1_1, buf18, arg20_1, arg19_1.item(), buf20, 4, 2048]
        self._run_test(
            "test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7",
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7.run,
            kernel_args,
            output_indices=[5],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_mul_silu_8(self, device):
        buf22 = rand_strided((4, 128), (128, 1), device=device, dtype=torch.bfloat16)
        buf23 = rand_strided(
            (4, 1, 128), (128, 128, 1), device=device, dtype=torch.bfloat16
        )
        # def triton_poi_fused_mul_silu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK)
        kernel_args = [buf23, buf22, 512]
        self._run_test(
            "test_triton_poi_fused_mul_silu_8",
            triton_poi_fused_mul_silu_8.run,
            kernel_args,
            output_indices=[],
            in_out_indices=[0],
            stream=self.stream0,
        )

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10(
        self, device
    ):
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
        # def triton_red_fused__..._rsqrt_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK, R0_BLOCK)
        kernel_args = [
            arg0_1,
            arg1_1,
            buf18,
            buf24,
            arg25_1,
            arg24_1.item(),
            buf26,
            4,
            2048,
        ]
        self._run_test(
            "test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10",
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10.run,
            kernel_args,
            output_indices=[6],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9(self, device):
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
        # def triton_red_fused__..._rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK, R0_BLOCK)
        # NOTE: The provided comment had 7 inputs, but the call has 8. Assuming in_ptr6 is arg34_1.item().
        kernel_args = [
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
        ]
        self._run_test(
            "test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9",
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9.run,
            kernel_args,
            output_indices=[7],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11(self, device):
        arg39_1 = torch.randn((), device="cpu", dtype=torch.float64)
        arg40_1 = rand_strided((2048,), (1,), device=device, dtype=torch.bfloat16)
        buf48 = rand_strided((4, 2048), (2048, 1), device=device, dtype=torch.bfloat16)
        buf50 = rand_strided(
            (4, 1, 2048), (2048, 2048, 1), device=device, dtype=torch.bfloat16
        )
        # def triton_red_fused__..._rsqrt_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK, R0_BLOCK)
        kernel_args = [buf50, buf48, arg40_1, arg39_1.item(), 512, 2048]
        self._run_test(
            "test_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11",
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run,
            kernel_args,
            output_indices=[],
            in_out_indices=[0],
            stream=self.stream0,
        )


instantiate_device_type_tests(
    TestCompiledLlamaOps, globals(), only_for=("xpu"), allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
