# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Unit tests for the MemFabric activation-offload backend (fsdp2).

The memfabric_hybrid / torch_npu packages are faked so the tests exercise the
allocator, registration refcounts and the MFSwapTensor state machine without
any NPU or MemFabric environment.
"""

import sys
import threading
import types
import unittest
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

from mindspeed_llm.fsdp2.features.async_offload import (  # pylint: disable=wrong-import-position
    MF_SLOT_ALIGN,
    MemFabricPool,
    MFSwapTensor,
    OffloadManager,
    PoolAllocator,
    SingletonMeta,
    _OffloadTask,
    _OffloadWorker,
    async_save_on_cpu,
)


class TestPoolAllocator(unittest.TestCase):
    def test_alloc_aligned_and_reuse(self):
        alloc = PoolAllocator()
        alloc.add_block(0x1000000, 0x100000)  # 1MiB block

        slot_a = alloc.alloc(1000)
        self.assertEqual(slot_a, 0x1000000)
        self.assertEqual(slot_a % MF_SLOT_ALIGN, 0)

        slot_b = alloc.alloc(4096)
        self.assertEqual(slot_b, 0x1001000)

        # exhausting returns None
        self.assertIsNone(PoolAllocator().alloc(1))

        # free and reuse (merged back into one range)
        alloc.free(slot_a)
        alloc.free(slot_b)
        slot_c = alloc.alloc(8192)
        self.assertEqual(slot_c, 0x1000000)
        self.assertEqual(alloc.total_size, 0x100000)

    def test_free_merges_adjacent_ranges(self):
        alloc = PoolAllocator()
        alloc.add_block(0, 6 * MF_SLOT_ALIGN)
        slots = [alloc.alloc(MF_SLOT_ALIGN) for _ in range(6)]
        for slot in slots:
            alloc.free(slot)
        # after full free the allocator must serve one big slot again
        big = alloc.alloc(6 * MF_SLOT_ALIGN)
        self.assertEqual(big, 0)
        self.assertEqual(alloc.used_size, 6 * MF_SLOT_ALIGN)

    def test_unaligned_block_start_is_aligned_up(self):
        alloc = PoolAllocator()
        alloc.add_block(0x1010, 0x10000)
        slot = alloc.alloc(64)
        self.assertEqual(slot % MF_SLOT_ALIGN, 0)
        self.assertGreaterEqual(slot, 0x1010)


class _FakeRallocHandle:
    def __init__(self, block_bytes):
        self._block_bytes = block_bytes
        self._next_gva = 0x100000000
        self.extend_calls = []
        self.register_calls = []
        self.unregister_calls = []
        self.copy_calls = []
        self.destroyed = False

    def extend_remote_mem(self, mem_type, size):
        self.extend_calls.append((mem_type, size))
        gva = self._next_gva
        self._next_gva += size
        return 0, {"rank_id": 3, "gva": gva}

    def register(self, ptr, size):
        self.register_calls.append((ptr, size))
        return 0

    def unregister(self, ptr):
        self.unregister_calls.append(ptr)
        return 0

    def copy_data(self, src, dst, size, flags):
        self.copy_calls.append((src, dst, size, flags))
        return 0

    def destroy(self):
        self.destroyed = True


class _FakeRalloc:
    def __init__(self, handle):
        self._handle = handle
        self.init_calls = []
        self.cfg = None
        self.create_kwargs = None

        self.RallocConfig = type(
            "RallocConfig", (), {"__init__": lambda self: None, "set_nic": lambda self, nic: None}
        )
        self.RallocRole = SimpleNamespace(NEAR=0, FAR=1)
        self.RallocMemType = SimpleNamespace(HOST=0, DEVICE=1)
        self.RallocDataOpType = SimpleNamespace(DEVICE_RDMA=8, HOST_RDMA=2, SDMA=1)
        self.rank_id = 11

    def initialize(self, store_url, world_size, device_id, cfg):
        self.init_calls.append((store_url, world_size, device_id, cfg))
        self.cfg = cfg
        return 0

    def get_rank_id(self):
        return self.rank_id

    def uninitialize(self, flags=0):
        return 0

    def create(self, id, max_dram_size, data_op_type=0, **kwargs):
        self.create_kwargs = {"id": id, "max_dram_size": max_dram_size, "data_op_type": data_op_type}
        return self._handle


def _install_fake_memfabric(handle):
    fake_ralloc = _FakeRalloc(handle)
    fake_mf = types.ModuleType("memfabric_hybrid")
    fake_mf.initialize = lambda flags=0: 0
    fake_mf.uninitialize = lambda flags=0: None
    fake_mf.get_last_err_msg = lambda: ""
    fake_ralloc_mod = types.ModuleType("memfabric_hybrid.ralloc")
    for attr in (
        "RallocConfig",
        "RallocRole",
        "RallocMemType",
        "RallocDataOpType",
        "initialize",
        "get_rank_id",
        "uninitialize",
        "create",
    ):
        setattr(fake_ralloc_mod, attr, getattr(fake_ralloc, attr))
    fake_mf.ralloc = fake_ralloc_mod
    sys.modules["memfabric_hybrid"] = fake_mf
    sys.modules["memfabric_hybrid.ralloc"] = fake_ralloc_mod
    return fake_mf, fake_ralloc


def _make_pool_args(**overrides):
    defaults = dict(
        mf_store_url="tcp://127.0.0.1:8572",
        mf_store_wait_timeout=5,
        mf_nic="tcp://127.0.0.1:10005",
        mf_world_size=16,
        offload_pool_size_gb=2,
        offload_extend_block_gb=1,
        offload_register_mode="per_tensor",
        mf_pool_id=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestMemFabricPool(unittest.TestCase):
    def setUp(self):
        SingletonMeta._instances.pop(MemFabricPool, None)
        SingletonMeta._instances.pop(OffloadManager, None)
        self.handle = _FakeRallocHandle(1 << 30)

    def tearDown(self):
        pool = SingletonMeta._instances.get(MemFabricPool)
        if pool is not None and pool.inited:
            pool.worker.stop()
        SingletonMeta._instances.pop(MemFabricPool, None)
        SingletonMeta._instances.pop(OffloadManager, None)
        sys.modules.pop("memfabric_hybrid", None)
        sys.modules.pop("memfabric_hybrid.ralloc", None)

    def test_initialize_near_role_auto_rank_and_block_table(self):
        _, fake_ralloc = _install_fake_memfabric(self.handle)
        pool = MemFabricPool()
        with mock.patch("mindspeed_llm.fsdp2.features.async_offload.mf_wait_for_store"):
            pool.initialize(_make_pool_args(), torch_rank=5, device_id=3)

        self.assertTrue(pool.inited)
        # NEAR role, store NOT started on training side, auto ranking
        self.assertEqual(fake_ralloc.cfg.role, fake_ralloc.RallocRole.NEAR)
        self.assertFalse(fake_ralloc.cfg.start_store)
        self.assertTrue(fake_ralloc.cfg.auto_ranking)
        self.assertEqual(fake_ralloc.create_kwargs["id"], 0)
        self.assertEqual(fake_ralloc.create_kwargs["data_op_type"], fake_ralloc.RallocDataOpType.DEVICE_RDMA)
        # auto-assigned ralloc rank recorded, not the torch rank
        self.assertEqual(pool.ralloc_rank, 11)
        # 2GB pool acquired as two 1GB blocks, contributor rank recorded
        self.assertEqual(len(self.handle.extend_calls), 2)
        self.assertEqual(len(pool.block_table), 2)
        self.assertEqual(pool.block_table[0]["contributor_rank"], 3)
        self.assertEqual(pool.allocator.total_size, 2 << 30)
        # worker thread started
        self.assertIsNotNone(pool.worker._thread)
        pool.worker.stop()

    def test_register_refcount(self):
        _install_fake_memfabric(self.handle)
        pool = MemFabricPool()
        with mock.patch("mindspeed_llm.fsdp2.features.async_offload.mf_wait_for_store"):
            pool.initialize(_make_pool_args())

        pool.register_dev(0x2000, 4096)
        pool.register_dev(0x2000, 4096)  # refcount hit, no duplicate register
        self.assertEqual(len(self.handle.register_calls), 1)
        pool.unregister_dev(0x2000)
        self.assertEqual(len(self.handle.unregister_calls), 0)  # still referenced
        pool.unregister_dev(0x2000)
        self.assertEqual(len(self.handle.unregister_calls), 1)

        pool.register_mode = "none"
        pool.register_dev(0x3000, 4096)
        pool.unregister_dev(0x3000)
        self.assertEqual(len(self.handle.register_calls), 1)
        pool.worker.stop()

    def test_copy_and_slot_lifecycle(self):
        _install_fake_memfabric(self.handle)
        pool = MemFabricPool()
        with mock.patch("mindspeed_llm.fsdp2.features.async_offload.mf_wait_for_store"):
            pool.initialize(_make_pool_args())

        gva = pool.alloc_slot(1000)
        self.assertIsNotNone(gva)
        used = pool.allocator.used_size
        pool.copy_l2g(0xdead0000, gva, 1000)
        pool.copy_g2l(gva, 0xdead0000, 1000)
        self.assertEqual(self.handle.copy_calls, [(0xdead0000, gva, 1000, 0), (gva, 0xdead0000, 1000, 0)])
        pool.free_slot(gva)
        self.assertEqual(pool.allocator.used_size, used - 4096)
        pool.worker.stop()

    def test_destroy(self):
        _install_fake_memfabric(self.handle)
        pool = MemFabricPool()
        with mock.patch("mindspeed_llm.fsdp2.features.async_offload.mf_wait_for_store"):
            pool.initialize(_make_pool_args())
        pool.destroy()
        self.assertFalse(pool.inited)
        self.assertTrue(self.handle.destroyed)
        # destroy is idempotent
        pool.destroy()


class TestMFSwapTensor(unittest.TestCase):
    def setUp(self):
        SingletonMeta._instances.pop(MemFabricPool, None)
        SingletonMeta._instances.pop(OffloadManager, None)
        self.handle = _FakeRallocHandle(1 << 30)
        _install_fake_memfabric(self.handle)
        self.pool = MemFabricPool()
        with mock.patch("mindspeed_llm.fsdp2.features.async_offload.mf_wait_for_store"):
            self.pool.initialize(_make_pool_args())
        # drive copies synchronously for determinism
        self.pool.worker.stop()
        self.worker = _OffloadWorker()
        self.worker.start()
        self.pool.worker = self.worker

    def tearDown(self):
        self.worker.stop()
        SingletonMeta._instances.pop(MemFabricPool, None)
        SingletonMeta._instances.pop(OffloadManager, None)
        sys.modules.pop("memfabric_hybrid", None)
        sys.modules.pop("memfabric_hybrid.ralloc", None)

    def test_full_d2h_h2d_lifecycle(self):
        tensor = torch.zeros(1024, dtype=torch.float32)
        swap = MFSwapTensor(tensor, "0_0", self.pool, register_mode="per_tensor", offload_enabled=True)

        original_ptr = tensor.data_ptr()
        swap.launch_d2h()
        self.assertEqual(self.pool.allocator.used_size, 4096)
        swap.wait_d2h_finished()
        self.assertEqual(swap.stat, "host")
        self.assertEqual(tensor.storage().size(), 0)  # device memory released

        swap.launch_h2d()
        self.assertEqual(swap.stat, "device")
        self.assertEqual(tensor.storage().size(), tensor.numel())  # restored
        self.assertEqual(self.pool.allocator.used_size, 0)  # slot recycled
        copies = self.handle.copy_calls
        self.assertEqual(len(copies), 2)
        self.assertEqual(copies[0][0], original_ptr)  # d2h from device ptr
        self.assertEqual(copies[0][1], 0x100000000)  # into the first pool slot
        self.assertEqual(copies[0][2], 4096)
        self.assertEqual(copies[1][0], 0x100000000)  # g2l back into restored tensor
        # registration paired around each copy
        self.assertEqual(len(self.handle.register_calls), 2)
        self.assertEqual(len(self.handle.unregister_calls), 2)

    def test_disabled_offload_keeps_tensor_intact(self):
        tensor = torch.ones(256, dtype=torch.float32)
        swap = MFSwapTensor(tensor, "last_0", self.pool, register_mode="per_tensor", offload_enabled=False)
        swap.launch_d2h()
        swap.wait_d2h_finished()
        self.assertEqual(tensor.storage().size(), 256)
        self.assertEqual(self.handle.copy_calls, [])
        swap.launch_h2d()
        self.assertEqual(tensor.storage().size(), 256)

    def test_pool_exhaustion_degrades_gracefully(self):
        tensor = torch.zeros(1 << 20, dtype=torch.uint8)
        swap = MFSwapTensor(tensor, "0_0", self.pool, register_mode="per_tensor", offload_enabled=True)
        # exhaust the pool
        while self.pool.alloc_slot(1 << 20) is not None:
            pass
        swap.launch_d2h()
        self.assertFalse(swap.offload_enabled)
        self.assertIsNone(swap.gva)
        swap.wait_d2h_finished()
        self.assertEqual(tensor.storage().size(), tensor.numel())

    def test_prefetch_h2d_on_worker(self):
        tensor = torch.zeros(2048, dtype=torch.float32)
        swap = MFSwapTensor(tensor, "1_0", self.pool, register_mode="per_tensor", offload_enabled=True)
        swap.launch_d2h()
        swap.wait_d2h_finished()
        swap.prefetch_launch_h2d()
        self.assertEqual(swap.stat, "device")
        # unpack arriving after prefetch: launch_h2d must just wait the prefetch
        # job out (no second restore, no extra copy) - regression for the guard
        # order that previously returned without waiting
        swap.launch_h2d()
        self.assertEqual(self.pool.allocator.used_size, 0)
        self.assertEqual(len(self.handle.copy_calls), 2)
        self.assertEqual(tensor.storage().size(), tensor.numel())
        swap.wait_h2d_finished()  # idempotent no-op afterwards
        self.assertEqual(len(self.handle.copy_calls), 2)


class TestAsyncSaveOnCpuFactory(unittest.TestCase):
    def test_backend_dispatch(self):
        pinned = async_save_on_cpu(
            h2d_stream=object(), d2h_stream=object(), block_idx=0, depth=4, backend="pinned"
        )
        self.assertIsInstance(pinned, torch.autograd.graph.saved_tensors_hooks)
        # memfabric backend builds hooks without touching streams
        mf_hooks = async_save_on_cpu(
            h2d_stream=None, d2h_stream=None, block_idx=0, depth=4, backend="memfabric"
        )
        self.assertIsInstance(mf_hooks, torch.autograd.graph.saved_tensors_hooks)


class TestOffloadTask(unittest.TestCase):
    def test_task_exception_propagates_on_wait(self):
        def _boom():
            raise RuntimeError("copy failed")

        task = _OffloadTask(_boom)
        task.run()
        with self.assertRaises(RuntimeError):
            task.wait()

    def test_task_ok(self):
        done = threading.Event()

        def _ok():
            done.set()

        task = _OffloadTask(_ok)
        task.run()
        task.wait()
        self.assertTrue(done.is_set())


class TestParallelArgumentsValidation(unittest.TestCase):
    def _load(self):
        from mindspeed_llm.fsdp2.utils.arguments import ParallelArguments

        return ParallelArguments

    def test_memfabric_requires_store_and_nic(self):
        ParallelArguments = self._load()
        with self.assertRaises(ValueError):
            ParallelArguments(activation_offload=True, activation_offload_backend="memfabric")
        with self.assertRaises(ValueError):
            ParallelArguments(
                activation_offload=True, activation_offload_backend="memfabric", mf_store_url="tcp://1.2.3.4:8572"
            )

    def test_valid_configuration(self):
        ParallelArguments = self._load()
        args = ParallelArguments(
            activation_offload=True,
            activation_offload_backend="memfabric",
            mf_store_url="tcp://1.2.3.4:8572",
            mf_nic="tcp://1.2.3.4:10005",
        )
        self.assertEqual(args.offload_pool_size_gb, 8)
        self.assertEqual(args.offload_register_mode, "per_tensor")

    def test_invalid_sizes(self):
        ParallelArguments = self._load()
        common = dict(
            activation_offload=True,
            activation_offload_backend="memfabric",
            mf_store_url="tcp://1.2.3.4:8572",
            mf_nic="tcp://1.2.3.4:10005",
        )
        with self.assertRaises(ValueError):
            ParallelArguments(**common, offload_pool_size_gb=0)
        with self.assertRaises(ValueError):
            ParallelArguments(**common, offload_extend_block_gb=16)
        with self.assertRaises(ValueError):
            ParallelArguments(**common, mf_pool_id=100)

    def test_pinned_backend_needs_nothing(self):
        ParallelArguments = self._load()
        args = ParallelArguments(activation_offload=True, activation_offload_backend="pinned")
        self.assertTrue(args.activation_offload)


if __name__ == "__main__":
    unittest.main()
