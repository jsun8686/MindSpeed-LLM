"""Unit tests for the mcore (Megatron) generic activation-offload integration.

Covers:
- ActivationOffloadFeature argument registration and v1 scope validation
- the _activation_offload_context helper inside transformer_block_forward
  (backend dispatch, gating, pinned stream caching)
- relocation regression: the shared mechanism lives in
  mindspeed_llm/core/memory/async_offload.py

Heavy deps (megatron / mindspeed / torch_npu) are expected from the CI
environment; torch_npu gets a CPU-safe fake when missing.
"""
import sys
import types
import unittest
from argparse import ArgumentParser
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch


class _FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def _install_fake_torch_npu():
    fake = types.SimpleNamespace(npu=SimpleNamespace(Event=_FakeEvent))
    sys.modules.setdefault("torch_npu", fake)


_install_fake_torch_npu()

from mindspeed_llm.features_manager.memory.activation_offload_feature import (  # pylint: disable=wrong-import-position
    ActivationOffloadFeature,
)


def _base_args(**overrides):
    """A valid flat mcore args namespace for activation offload."""
    args = SimpleNamespace(
        activation_offload=True,
        activation_offload_backend="memfabric",
        offload_pool_size_gb=8,
        offload_extend_block_gb=1,
        offload_register_mode="per_tensor",
        mf_store_url="tcp://10.0.0.1:8572",
        mf_nic="tcp://10.0.0.1:10005",
        mf_world_size=64,
        mf_pool_id=0,
        mf_store_wait_timeout=300,
        pipeline_model_parallel_size=1,
        num_layers_per_virtual_pipeline_stage=None,
        context_parallel_size=1,
        share_kvstates=False,
        n_hash_layers=0,
        recompute_granularity=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestActivationOffloadFeatureArgs(unittest.TestCase):
    def setUp(self):
        self.feature = ActivationOffloadFeature()
        self.parser = ArgumentParser()
        self.feature.register_args(self.parser)

    def test_defaults(self):
        args = self.parser.parse_args([])
        self.assertFalse(args.activation_offload)
        self.assertEqual(args.activation_offload_backend, "memfabric")
        self.assertEqual(args.offload_pool_size_gb, 8)
        self.assertEqual(args.offload_extend_block_gb, 1)
        self.assertEqual(args.offload_register_mode, "per_tensor")
        self.assertIsNone(args.mf_store_url)
        self.assertIsNone(args.mf_nic)
        self.assertEqual(args.mf_world_size, 64)
        self.assertEqual(args.mf_pool_id, 0)
        self.assertEqual(args.mf_store_wait_timeout, 300)

    def test_full_command_line(self):
        args = self.parser.parse_args([
            "--activation-offload",
            "--activation-offload-backend", "pinned",
            "--offload-pool-size-gb", "4",
            "--offload-extend-block-gb", "2",
            "--offload-register-mode", "none",
            "--mf-store-url", "tcp://1.2.3.4:8572",
            "--mf-nic", "tcp://1.2.3.4:10005",
            "--mf-world-size", "32",
            "--mf-pool-id", "3",
            "--mf-store-wait-timeout", "60",
        ])
        self.assertTrue(args.activation_offload)
        self.assertEqual(args.activation_offload_backend, "pinned")
        self.assertEqual(args.offload_pool_size_gb, 4)
        self.assertEqual(args.offload_extend_block_gb, 2)
        self.assertEqual(args.offload_register_mode, "none")
        self.assertEqual(args.mf_world_size, 32)
        self.assertEqual(args.mf_pool_id, 3)
        self.assertEqual(args.mf_store_wait_timeout, 60)

    def test_invalid_backend_choice_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--activation-offload-backend", "hugepages"])


class TestActivationOffloadFeatureValidation(unittest.TestCase):
    def setUp(self):
        self.feature = ActivationOffloadFeature()

    def test_valid_memfabric_passes(self):
        self.feature.validate_args(_base_args())

    def test_valid_pinned_passes_without_mf_args(self):
        self.feature.validate_args(_base_args(activation_offload_backend="pinned", mf_store_url=None, mf_nic=None))

    def test_noop_when_disabled(self):
        # validation must not fire for unrelated misconfigurations when off
        self.feature.validate_args(_base_args(activation_offload=False, pipeline_model_parallel_size=8))

    def test_memfabric_requires_store_url(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(mf_store_url=None))

    def test_memfabric_requires_nic(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(mf_nic=None))

    def test_pool_size_must_be_positive(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(offload_pool_size_gb=0))

    def test_extend_block_bounds(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(offload_extend_block_gb=0))
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(offload_extend_block_gb=9))

    def test_reject_pp(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(pipeline_model_parallel_size=2))

    def test_reject_vpp(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(num_layers_per_virtual_pipeline_stage=4))

    def test_reject_cp(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(context_parallel_size=2))

    def test_reject_share_kvstates(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(share_kvstates=True))

    def test_reject_hash_layers(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(n_hash_layers=2))

    def test_reject_full_recompute(self):
        with self.assertRaises(AssertionError):
            self.feature.validate_args(_base_args(recompute_granularity="full"))


class TestTransformerBlockOffloadContext(unittest.TestCase):
    """Tests for the _activation_offload_context mounting helper."""

    MODULE = "mindspeed_llm.core.transformer.transformer_block"

    def _helper(self):
        # imported lazily so this suite is skipped-fast when megatron is absent
        module = __import__(self.MODULE, fromlist=["_activation_offload_context"])
        return module._activation_offload_context

    def _block(self, training=True):
        return SimpleNamespace(training=training, layers=[0, 0, 0, 0])

    def test_disabled_returns_nullcontext(self):
        helper = self._helper()
        args = _base_args(activation_offload=False)
        with mock.patch(f"{self.MODULE}.get_args", return_value=args):
            ctx = helper(self._block(), 0, torch.zeros(4))
            self.assertIsInstance(ctx, nullcontext)

    def test_eval_returns_nullcontext(self):
        helper = self._helper()
        with mock.patch(f"{self.MODULE}.get_args", return_value=_base_args()):
            ctx = helper(self._block(training=False), 0, torch.zeros(4))
            self.assertIsInstance(ctx, nullcontext)

    def test_memfabric_backend_dispatch(self):
        helper = self._helper()
        with mock.patch(f"{self.MODULE}.get_args", return_value=_base_args()):
            ctx = helper(self._block(), 2, torch.zeros(4))
            self.assertIsInstance(ctx, torch.autograd.graph.saved_tensors_hooks)

    def test_pinned_backend_caches_stream_on_block(self):
        helper = self._helper()
        args = _base_args(activation_offload_backend="pinned")
        block = self._block()
        with mock.patch(f"{self.MODULE}.get_args", return_value=args), \
                mock.patch(f"{self.MODULE}.torch") as fake_torch:
            fake_stream = fake_torch.npu.Stream.return_value
            ctx = helper(block, 0, torch.zeros(4))
            self.assertIsInstance(ctx, torch.autograd.graph.saved_tensors_hooks)
            self.assertEqual(fake_torch.npu.Stream.call_count, 1)
            self.assertIs(block._activation_offload_stream, fake_stream)
            # second layer reuses the cached stream
            helper(block, 1, torch.zeros(4))
            self.assertEqual(fake_torch.npu.Stream.call_count, 1)


class TestSharedModuleRelocation(unittest.TestCase):
    def test_mechanism_module_lives_in_core_memory(self):
        import mindspeed_llm.core.memory.async_offload as shared

        for symbol in ("MemFabricPool", "MFSwapTensor", "PoolAllocator", "async_save_on_cpu"):
            self.assertTrue(hasattr(shared, symbol), symbol)


if __name__ == "__main__":
    unittest.main()
