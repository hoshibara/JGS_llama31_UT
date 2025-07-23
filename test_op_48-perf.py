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

from op_48_perf import (
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

torch.manual_seed(42)
torch.xpu.manual_seed(42)


class TestCompiledLlama48PerfOps(TestCase):
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
    UT_MODE = os.environ.get("UT_MODE", "COMPARE").upper()
    DATA_DIR = pathlib.Path(__file__).parent / "48_perf_test_data"

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
        self,
        test_name,
        func,
        kernel_args_list,
        output_indices,
        in_out_indices,
        stream,
        device="xpu",
        **kwargs,
    ):
        """Helper function to handle DUMP and COMPARE logic."""
        input_file = self.DATA_DIR / f"{test_name}_input.pt"
        output_file = self.DATA_DIR / f"{test_name}_output.pt"

        if self.UT_MODE == "DUMP":
            print(f"[DUMP] Running '{test_name}' and saving data...")

            # Save all initial arguments
            # Note: For in_out_ptr, this saves the initial state.
            # torch.save({"args_list": kernel_args_list, "kwargs": kwargs}, input_file)

            # Run the kernel
            # print('[DEBUG][DUMP] Running kernel with args:', kernel_args)
            # print('[DEBUG][DUMP] Running kernel with kwargs:', kwargs)
            out_tensors_list = []
            in_out_tensors_list = []
            for kernel_args in kernel_args_list:
                func(*kernel_args, stream=stream, **kwargs)
                torch.xpu.synchronize()
                out_tensors_list.append(
                    [kernel_args[i].clone() for i in output_indices]
                )
                in_out_tensors_list.append(
                    [kernel_args[i].clone() for i in in_out_indices]
                )

            # Collect output tensors
            outputs_to_save = {
                "out_tensors_list": out_tensors_list,
                "in_out_tensors_list": in_out_tensors_list,
            }
            torch.save(outputs_to_save, output_file)
            self.assertTrue(True)

        else:  # COMPARE mode
            print(f"[COMPARE] Running '{test_name}' and verifying results...")
            if not output_file.exists():
                self.fail(
                    f"Data files for '{test_name}' not found. Run in DUMP mode first."
                )

            # Load inputs and golden outputs
            # loaded_data = torch.load(input_file, map_location=device)
            # kernel_args_list = loaded_data["args_list"]
            # kwargs = loaded_data["kwargs"]

            golden_outputs = torch.load(output_file, map_location=device)
            golden_out_tensors_list = golden_outputs["out_tensors_list"]
            golden_in_out_tensors_list = golden_outputs["in_out_tensors_list"]

            for kernel_args, golden_out_tensors, golden_in_out_tensors in zip(
                kernel_args_list, golden_out_tensors_list, golden_in_out_tensors_list
            ):
                # Run the kernel with loaded inputs
                # print('[DEBUG][COMPARE] Running kernel with args:', run_args)
                # print('[DEBUG][COMPARE] Running kernel with kwargs:', kwargs)
                func(*kernel_args, stream=stream, **kwargs)
                torch.xpu.synchronize()

                # Compare results
                for i, idx in enumerate(output_indices):
                    self.assertEqual(
                        kernel_args[idx],
                        golden_out_tensors[i],
                        msg=f"Output mismatch at index {idx}",
                    )

                for i, idx in enumerate(in_out_indices):
                    self.assertEqual(
                        kernel_args[idx],
                        golden_in_out_tensors[i],
                        msg=f"In-out mismatch at index {idx}",
                    )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3(
        self, device
    ):
        buf2 = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        buf4 = torch.randn((4, 64, 1024), device=device, dtype=torch.float32)
        arg6_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf8 = torch.randn((4, 32, 1024, 128), device=device, dtype=torch.bfloat16)
        kernel_args = [[buf2, buf4, arg6_1.item(), buf8, 16777216]]
        self._run_test(
            "triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_3",
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
            (4, 4, 1024, 128),
            (524288, 128, 512, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        buf9 = torch.randn((4, 32, 1024, 128), device=device, dtype=torch.bfloat16)
        buf7 = torch.randn((4096, 512), device=device, dtype=torch.bfloat16)
        buf10 = torch.randn((4, 32, 1024, 128), device=device, dtype=torch.bfloat16)
        kernel_args = [
            [buf6, buf9, 16777216],
            [buf7, buf10, 16777216],
        ]
        self._run_test(
            "triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_4.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5(
        self, device
    ):
        arg2_1 = torch.zeros((1024,), device=device, dtype=torch.int64)
        arg4_1 = rand_strided((4, 1024), (0, 1), device=device, dtype=torch.int64)
        buf11 = torch.randn((4, 1, 1024, 1024), device=device, dtype=torch.bfloat16)
        buf34 = torch.randn((4, 1, 1024, 1024), device=device, dtype=torch.bfloat16)
        kernel_args = [[arg2_1, arg4_1, buf11, buf34, 4194304]]
        self._run_test(
            "triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_clone_mul_scalar_tensor_where_5.run,
            kernel_args,
            output_indices=[2, 3],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_add_cat_mul_2(self, device):
        buf5 = torch.randn((4096, 512), device=device, dtype=torch.bfloat16)
        buf4 = torch.randn((4, 64, 1024), device=device, dtype=torch.float32)
        arg6_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf6 = rand_strided(
            (4, 4, 1024, 128),
            (524288, 128, 512, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        kernel_args = [[buf5, buf4, arg6_1.item(), buf6, 4096, 512]]
        self._run_test(
            "triton_poi_fused_add_cat_mul_2",
            triton_poi_fused_add_cat_mul_2.run,
            kernel_args,
            output_indices=[3],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_clone_6(self, device):
        buf13 = torch.randn((4, 32, 1024, 128), device=device, dtype=torch.bfloat16)
        buf17 = torch.randn((4, 1024, 32, 128), device=device, dtype=torch.bfloat16)
        kernel_args = [[buf13, buf17, 16777216]]
        self._run_test(
            "triton_poi_fused_clone_6",
            triton_poi_fused_clone_6.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    @unittest.skip("IOT instruction")
    def test_triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0(self, device):
        kernel_args = [
            [
                torch.zeros((4, 1), device=device, dtype=torch.int64),
                torch.randn((128256, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                torch.randn((4, 1, 4096), device=device, dtype=torch.bfloat16),
                4,
                4096,
            ],
            [
                torch.zeros((4, 1024), device=device, dtype=torch.int64),
                torch.randn((128256, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                torch.randn((4, 1024, 4096), device=device, dtype=torch.bfloat16),
                4096,
                4096,
            ],
        ]
        self._run_test(
            "triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0",
            triton_red_fused__to_copy_embedding_mean_mul_pow_rsqrt_0.run,
            kernel_args,
            output_indices=[4],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__to_copy_1(self, device):
        arg3_1 = torch.zeros((4, 1), device=device, dtype=torch.int64)
        buf3 = rand_strided((4, 1, 1), (1, 4, 4), device=device, dtype=torch.float32)
        arg3_1_2 = torch.zeros((4, 1024), device=device, dtype=torch.int64)
        buf3_2 = rand_strided(
            (4, 1, 1024), (1024, 4096, 1), device=device, dtype=torch.float32
        )
        kernel_args = [
            [arg3_1, buf3, 4],
            [arg3_1_2, buf3_2, 4096],
        ]
        self._run_test(
            "triton_poi_fused__to_copy_1",
            triton_poi_fused__to_copy_1.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_cat_2(self, device):
        arg6_1 = rand_strided(
            (4, 4, 1024, 128),
            (524288, 128, 512, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        buf5 = torch.randn((4, 512), device=device, dtype=torch.bfloat16)
        buf4 = torch.randn((4, 64, 1), device=device, dtype=torch.float32)
        arg8_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf6 = torch.randn((4, 4, 1025, 128), device=device, dtype=torch.bfloat16)
        kernel_args = [
            [arg6_1, buf5, buf4, arg8_1.item(), buf6, 2099200],
        ]
        self._run_test(
            "triton_poi_fused_cat_2",
            triton_poi_fused_cat_2.run,
            kernel_args,
            output_indices=[4],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_cat_3(self, device):
        arg15_1 = rand_strided(
            (4, 4, 1024, 128),
            (524288, 128, 512, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        buf7 = torch.randn((4, 512), device=device, dtype=torch.bfloat16)
        buf8 = torch.randn((4, 4, 1025, 128), device=device, dtype=torch.bfloat16)
        kernel_args = [
            [
                arg15_1,
                buf7,
                buf8,
                2099200,
            ]
        ]
        self._run_test(
            "triton_poi_fused_cat_3",
            triton_poi_fused_cat_3.run,
            kernel_args,
            output_indices=[2],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4(
        self, device
    ):
        buf2 = torch.randn((4, 4096), device=device, dtype=torch.bfloat16)
        buf4 = torch.randn((4, 64, 1), device=device, dtype=torch.float32)
        arg9_1 = torch.randn((), device="cpu", dtype=torch.float64)
        buf9 = torch.randn((4, 32, 1, 128), device=device, dtype=torch.bfloat16)
        kernel_args = [[buf2, buf4, arg9_1.item(), buf9, 16384]]
        self._run_test(
            "triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_4.run,
            kernel_args,
            output_indices=[3],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5(
        self, device
    ):
        buf6 = torch.randn((4, 4, 1025, 128), device=device, dtype=torch.bfloat16)
        buf10 = torch.randn((4, 32, 1025, 128), device=device, dtype=torch.bfloat16)
        kernel_args = [[buf6, buf10, 16793600]]
        self._run_test(
            "triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_5.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_6(
        self, device
    ):
        arg2_1 = torch.zeros((1,), device=device, dtype=torch.int64)
        arg5_1 = torch.zeros((4, 1025), device=device, dtype=torch.int64)
        buf12 = torch.randn((4, 1, 1, 1025), device=device, dtype=torch.bfloat16)
        buf35 = torch.randn((4, 1, 1, 1025), device=device, dtype=torch.bfloat16)
        kernel_args = [
            [arg2_1, arg5_1, buf12, buf35, 1025, 4100],
        ]
        self._run_test(
            "triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_6",
            triton_poi_fused__scaled_dot_product_fused_attention_overrideable_add_cat_mul_scalar_tensor_where_6.run,
            kernel_args,
            output_indices=[1],
            in_out_indices=[],
            stream=self.stream0,
        )

    @unittest.skip("IOT instruction")
    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7(self, device):
        kernel_args = [
            [
                torch.zeros((4, 1), device=device, dtype=torch.int64),
                torch.randn((128256, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                torch.randn((4, 1, 4096), device=device, dtype=torch.bfloat16),
                4,
                4096,
            ],
            [
                torch.zeros((4, 1024), device=device, dtype=torch.int64),
                torch.randn((128256, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                torch.randn((4, 1, 4096), device=device, dtype=torch.bfloat16),
                4096,
                4096,
            ],
        ]
        self._run_test(
            "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7",
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_7.run,
            kernel_args,
            output_indices=[5],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_poi_fused_mul_silu_8(self, device):
        buf47 = torch.randn((4, 1, 2048), device=device, dtype=torch.bfloat16)
        buf46 = torch.randn((4, 2048), device=device, dtype=torch.bfloat16)
        buf23 = torch.randn((4, 1024, 2048), device=device, dtype=torch.bfloat16)
        buf22 = torch.randn((4096, 2048), device=device, dtype=torch.bfloat16)
        kernel_args = [
            [buf47, buf46, 8192],
            [buf23, buf22, 8388608],
        ]
        self._run_test(
            "triton_poi_fused_mul_silu_8",
            triton_poi_fused_mul_silu_8.run,
            kernel_args,
            output_indices=[],
            in_out_indices=[0],
            stream=self.stream0,
        )

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10(
        self, device
    ):
        kernel_args = [
            [
                torch.randn((4, 1, 4096), device=device, dtype=torch.bfloat16),
                torch.zeros((4, 1), device=device, dtype=torch.int64),
                torch.randn((128256, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                torch.randn((4, 1, 4096), device=device, dtype=torch.bfloat16),
                4,
                4096,
            ],
            [
                torch.randn((4, 1024, 4096), device=device, dtype=torch.bfloat16),
                torch.zeros((4, 1024), device=device, dtype=torch.int64),
                torch.randn((128256, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                torch.randn((4, 1024, 4096), device=device, dtype=torch.bfloat16),
                4096,
                4096,
            ],
        ]
        self._run_test(
            "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10",
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_10.run,
            kernel_args,
            output_indices=[7],
            in_out_indices=[0],
            stream=self.stream0,
        )

    def test_triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9(self, device):
        kernel_args = [
            [
                torch.zeros((4, 1), device=device, dtype=torch.int64),
                torch.randn((128256, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                torch.randn((4, 1, 4096), device=device, dtype=torch.bfloat16),
                4,
                4096,
            ],
            [
                torch.zeros((4, 1024), device=device, dtype=torch.int64),
                torch.randn((128256, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                torch.randn((4, 1024, 4096), device=device, dtype=torch.bfloat16),
                4096,
                4096,
            ],
        ]
        self._run_test(
            "triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9",
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9.run,
            kernel_args,
            output_indices=[6],
            in_out_indices=[],
            stream=self.stream0,
        )

    def test_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11(self, device):
        kernel_args = [
            [
                torch.randn((4, 1, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                4,
                4096,
            ],
            [
                torch.randn((4, 1024, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096, 4096), device=device, dtype=torch.bfloat16),
                torch.randn((4096,), device=device, dtype=torch.bfloat16),
                torch.randn((), device="cpu", dtype=torch.float64).item(),
                4096,
                4096,
            ],
        ]
        self._run_test(
            "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11",
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_11.run,
            kernel_args,
            output_indices=[],
            in_out_indices=[0],
            stream=self.stream0,
        )


instantiate_device_type_tests(
    TestCompiledLlama48PerfOps, globals(), only_for=("xpu"), allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
