import queue
import socket
import threading
import time

import torch
try:
    import torch_npu
except ImportError:
    pass
from torch.autograd.graph import saved_tensors_hooks

from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)

# Slot alignment inside acquired pool blocks (device RDMA / MR registration granularity).
MF_SLOT_ALIGN = 4096
# Block granularity required by ralloc extend (create2/extend sizes must be 2M aligned).
MF_BLOCK_ALIGN = 2 * 1024 * 1024


def _align_up(value, align):
    return (value + align - 1) // align * align


def mf_wait_for_store(store_url, timeout_sec=300.0, poll_interval=0.5):
    """Block until the config store at store_url (tcp://ip:port) is reachable.

    The ralloc store is hosted on the FAR side and must be up before a NEAR
    (training) process calls ralloc.initialize for master discovery.
    """
    try:
        host, port = store_url.split("://", 1)[-1].rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid store url: {store_url}, expected tcp://ip:port") from exc
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex((host, int(port))) == 0:
                return
        time.sleep(poll_interval)
    raise RuntimeError(f"config store not reachable within {timeout_sec}s: {store_url}")


def base_check_fn(tensor):
    if isinstance(tensor._base, torch.nn.parameter.Parameter) or isinstance(tensor, torch.nn.parameter.Parameter):
        return False
    if tensor.storage().size() <= 0:
        return False
    return True


class GetCnt:
    def __init__(self):
        self._block_idx = -1
        self._block_tensor_nums = {}  # offload tensors per block

    def get_cnt(self, block_idx):
        after_block = False
        if block_idx > self._block_idx:
            self._block_tensor_nums[block_idx] = 1
            if block_idx != 0:
                after_block = True
            self._block_idx = block_idx
        elif block_idx == self._block_idx:
            self._block_tensor_nums[block_idx] += 1
        else:
            # one step end
            self._block_idx = block_idx
            self._block_tensor_nums = {block_idx: 1}

        offload_tensor_key = "{}_{}".format(self._block_idx, self._block_tensor_nums[self._block_idx] - 1)
        return offload_tensor_key, after_block

    def get_prefetch_keys(self, block_idx, tensor_idx):
        prefetch_block_idx = max((idx for idx in self._block_tensor_nums.keys() if idx < block_idx), default=None)

        if prefetch_block_idx is None:
            return []

        prefetch_block_tensor_nums = self._block_tensor_nums[prefetch_block_idx]
        block_tensor_nums = self._block_tensor_nums[block_idx]
        start = tensor_idx * prefetch_block_tensor_nums // block_tensor_nums
        end = (tensor_idx + 1) * prefetch_block_tensor_nums // block_tensor_nums
        prefetch_idxs = list(range(start, end))
        return ["{}_{}".format(block_idx - 1, prefetch_idx) for prefetch_idx in prefetch_idxs]


class SwapTensor:
    def __init__(self, tensor, key):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device='cpu')

        self.is_slice_tensor = tensor.storage().size() != tensor.numel()
        self.stat = "device"
        self.key = key

        self.h2d_event = torch_npu.npu.Event()

    # device to host
    def launch_d2h(self, stream):
        if self.stat != "device":
            return

        forward_event = torch_npu.npu.Event()
        forward_event.record()
        with torch.no_grad():
            with torch_npu.npu.stream(stream):
                stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.tensor_cpu.copy_(self.tensor, non_blocking=True)
                else:
                    self.tensor_cpu.storage().copy_(self.tensor.storage(), non_blocking=True)
                self.stat = "host"

    # synchronize d2h and resize 0
    def wait_d2h_finished(self, stream, flag):
        if self.stat != "host":
            return
        if flag:
            torch_npu.npu.current_stream().wait_stream(stream)
            torch_npu.npu.default_stream().wait_stream(stream)
        self.tensor.storage().resize_(0)
        self.stat = "host"

    # resize storage_size and host to device
    def launch_h2d(self, h2d_stream, flag, working_stream):
        if self.stat != "host":
            return
        backward_event = torch_npu.npu.Event()
        backward_event.record()
        if flag:
            self.tensor.storage().resize_(self.storage_size)
        with torch.no_grad():
            with torch_npu.npu.stream(h2d_stream):
                h2d_stream.wait_event(backward_event)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "device"

                working_stream.wait_stream(h2d_stream)

    # resize storage_size and host to device
    def prefetch_launch_h2d(self, h2dstream, flag):
        if self.stat != "host":
            return
        backward_event = torch_npu.npu.Event()
        backward_event.record()
        if flag:
            self.tensor.storage().resize_(self.storage_size)
        with torch.no_grad():
            with torch_npu.npu.stream(h2dstream):
                h2dstream.wait_event(backward_event)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "device"
                self.tensor.record_stream(h2dstream)

    # synchronize h2d
    def wait_h2d_finished(self):
        if self.stat != "device":
            return
        if self.h2d_event:
            torch_npu.npu.current_stream().wait_event(self.h2d_event)
            torch_npu.npu.default_stream().wait_event(self.h2d_event)
        self.stat = "device"


class SingletonMeta(type):
    """
    single meta class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


class OffloadItem:
    """
    class for offload item
    """

    def __init__(self, act=None, ref_cnt=0, event=None):
        self.act = act
        self.ref_cnt = ref_cnt
        self.event = event

    def get_event(self):
        return self.event

    def has_event(self):
        return self.event is not None


class OffloadManager(metaclass=SingletonMeta):
    """
    class for offload manager
    """

    def __init__(self, check=False):
        self.items = {}
        self.check = check
        self.npu_item = []
        self.getcnt = GetCnt()

    def get_cnt(self, block_idx):
        return self.getcnt.get_cnt(block_idx)

    def assert_exist(self, key):
        if key not in self.items:
            raise RuntimeError(f"Key {key} does not exist in items")

    def exist(self, key):
        return key in self.items

    def assert_not_exist(self, key):
        if key not in self.items:
            raise RuntimeError(f"Key {key} already exist in items")

    def put(self, key, act, event=None):
        if key in self.items:
            self.items[key].act = act
            self.items[key].ref_cnt += 1
            self.items[key].event = event
        else:
            self.items[key] = OffloadItem(act, 1, event)

    def put_npu_tensor(self, act):
        self.npu_item.append(act)

    def del_npu_tensor(self, prefile_key, d2h_stream):
        for key in self.items.keys():
            if key.startswith(prefile_key):
                self.items[key].act.wait_d2h_finished(d2h_stream, True)

    def get(self, key):
        self.assert_exist(key)
        item = self.items[key]

        act = item.act
        if item.has_event():
            item.get_event().wait()

        item.ref_cnt -= 1
        if item.ref_cnt == 0:
            self.clear(key)
        return act

    def prefetch_get(self, block_idx, tensor_idx, h2d_stream, d2h_stream):
        prefetch_keys = self.getcnt.get_prefetch_keys(block_idx, tensor_idx)
        for prefetch_key in prefetch_keys:
            if self.exist(prefetch_key):
                prefetch_swap_tensor = self.get(prefetch_key)
                if h2d_stream is not None and d2h_stream is not None:
                    d2h_stream.wait_stream(h2d_stream)
                prefetch_swap_tensor.prefetch_launch_h2d(h2d_stream, True)
                if h2d_stream is not None and hasattr(prefetch_swap_tensor.tensor, "record_stream"):
                    prefetch_swap_tensor.tensor.record_stream(h2d_stream)

    def empty(self):
        return len(self.items) == 0

    def clear(self, key=None):
        if key is None:
            self.items.clear()
        else:
            self.assert_exist(key)
            self.items.pop(key)

    # event interface #

    def get_event(self, key):
        self.assert_exist(key)
        item = self.items[key]
        event = item.get_event()
        return event

    def has_event(self, key):
        if not self.exist(key):
            return False
        item = self.items[key]
        return item.has_event()


class PoolAllocator:
    """Fine-grained sub-allocator over GVA blocks acquired from the MemFabric pool.

    Training ranks request coarse blocks (2M aligned) via ralloc and manage
    fine-grained 4K aligned slots locally; ranks never need to know where the
    coarse blocks physically reside.
    """

    def __init__(self, slot_align=MF_SLOT_ALIGN):
        self.slot_align = slot_align
        self._free_ranges = []  # sorted list of [start, end)
        self._allocs = {}  # gva -> allocated (aligned) size

    def add_block(self, gva, size):
        self._insert_free(gva, gva + size)

    def alloc(self, nbytes):
        need = _align_up(max(nbytes, 1), self.slot_align)
        for idx, (start, end) in enumerate(self._free_ranges):
            slot = _align_up(start, self.slot_align)
            if slot + need <= end:
                self._allocs[slot] = need
                remainder_left = (slot, slot - start)
                remainder_right = (slot + need, end)
                del self._free_ranges[idx]
                if remainder_left[1] > remainder_left[0]:
                    self._insert_free(*remainder_left)
                if remainder_right[1] > remainder_right[0]:
                    self._insert_free(*remainder_right)
                return slot
        return None

    def free(self, gva):
        nbytes = self._allocs.pop(gva, None)
        if nbytes is None:
            return
        self._insert_free(gva, gva + nbytes)

    def _insert_free(self, start, end):
        import bisect

        idx = bisect.bisect_left(self._free_ranges, [start, end])
        # merge with left neighbour
        if idx > 0 and self._free_ranges[idx - 1][1] >= start:
            idx -= 1
            start = min(start, self._free_ranges[idx][0])
            end = max(end, self._free_ranges[idx][1])
            del self._free_ranges[idx]
        # merge with right neighbours
        while idx < len(self._free_ranges) and self._free_ranges[idx][0] <= end:
            end = max(end, self._free_ranges[idx][1])
            del self._free_ranges[idx]
        self._free_ranges.insert(idx, [start, end])

    @property
    def total_size(self):
        return sum(end - start for start, end in self._free_ranges) + sum(self._allocs.values())

    @property
    def used_size(self):
        return sum(self._allocs.values())


class _OffloadTask:
    """A unit of work executed on the offload worker thread."""

    __slots__ = ("fn", "done", "exc")

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.exc = None

    def run(self):
        try:
            self.fn()
        except BaseException as exc:  # pylint: disable=broad-except
            self.exc = exc
        finally:
            self.done.set()

    def wait(self):
        self.done.wait()
        if self.exc is not None:
            raise self.exc


class _OffloadWorker:
    """Single background thread digesting blocking MemFabric copies.

    DEVICE_RDMA copy_data is host-synchronous (DataCopyAsync is not supported
    by the device-rdma data path), so asynchrony against the training thread is
    achieved by running the copies on this worker, coordinated with NPU events
    recorded on the compute stream.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="mf-offload-worker", daemon=True)
            self._thread.start()

    def _loop(self):
        while True:
            task = self._queue.get()
            if task is None:
                return
            task.run()

    def submit(self, fn):
        task = _OffloadTask(fn)
        self._queue.put(task)
        return task

    def stop(self):
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            self._queue.put(None)
            thread.join(timeout=30)


class MemFabricPool(metaclass=SingletonMeta):
    """Per-rank MemFabric DRAM pool client (NEAR role) for activation offload.

    Lifecycle: FAR contributor daemons (one of them hosting the config store)
    are started first, then each training rank initializes with auto_ranking
    enabled, records the auto-assigned ralloc rank id, and acquires remote DRAM
    blocks via extend_remote_mem. The training framework rank id is never used
    to configure ralloc.
    """

    def __init__(self):
        self.inited = False
        self.mf = None
        self.ralloc = None
        self.handle = None
        self.ralloc_rank = None
        self.allocator = PoolAllocator()
        self.worker = _OffloadWorker()
        self.block_table = []  # [{"gva": int, "size": int, "contributor_rank": int}]
        self.register_mode = "per_tensor"
        self._reg_entries = {}  # ptr -> [refcnt, size]
        self._reg_lock = threading.Lock()
        self._alloc_fail_count = 0

    def initialize(self, args, torch_rank=0, device_id=0):
        """Create the pool. `args` is the flattened fsdp2 args namespace."""
        if self.inited:
            return
        try:
            import memfabric_hybrid as mf
            from memfabric_hybrid import ralloc
        except ImportError as exc:
            raise ImportError(
                "memfabric_hybrid is a hard dependency of the memfabric activation-offload "
                "backend. Install it with `pip install memfabric_hybrid` or switch "
                "--parallel.activation-offload-backend to 'pinned'."
            ) from exc

        store_url = args.mf_store_url
        logger.info_rank0(f"[MemFabricPool] waiting for FAR-hosted config store: {store_url}")
        mf_wait_for_store(store_url, timeout_sec=args.mf_store_wait_timeout)

        if mf.initialize() != 0:
            raise RuntimeError(f"memfabric_hybrid initialize failed: {mf.get_last_err_msg()}")
        self.mf = mf
        self.ralloc = ralloc

        cfg = ralloc.RallocConfig()
        cfg.role = ralloc.RallocRole.NEAR  # training ranks request memory
        cfg.start_store = False  # the store is hosted on the FAR side
        cfg.auto_ranking = True  # rank ids auto-assigned by ralloc
        if getattr(args, "mf_nic", None):
            cfg.set_nic(args.mf_nic)
        if ralloc.initialize(store_url, args.mf_world_size, device_id, cfg) != 0:
            raise RuntimeError(f"ralloc initialize failed: {mf.get_last_err_msg()}")

        # Record auto-assigned ralloc rank id (NOT the training framework rank id).
        self.ralloc_rank = ralloc.get_rank_id()
        logger.info(f"[MemFabricPool] torch_rank={torch_rank} -> ralloc_rank={self.ralloc_rank}")
        self._log_rank_mapping(torch_rank)

        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        total_bytes = int(args.offload_pool_size_gb) << 30
        block_bytes = _align_up(int(args.offload_extend_block_gb) << 30, MF_BLOCK_ALIGN)
        self.register_mode = getattr(args, "offload_register_mode", "per_tensor")
        handle = ralloc.create(
            id=int(getattr(args, "mf_pool_id", 0)),
            max_dram_size=total_bytes,
            data_op_type=ralloc.RallocDataOpType.DEVICE_RDMA,
        )
        if handle is None:
            raise RuntimeError(f"ralloc create failed: {mf.get_last_err_msg()}")

        deadline = time.time() + args.mf_store_wait_timeout
        remaining = total_bytes
        while remaining > 0:
            request = min(_align_up(remaining, MF_BLOCK_ALIGN), block_bytes)
            ret, info = handle.extend_remote_mem(ralloc.RallocMemType.HOST, request)
            if ret != 0:
                if time.time() > deadline:
                    raise RuntimeError(
                        f"extend_remote_mem failed within {args.mf_store_wait_timeout}s: "
                        f"ret={ret} err={mf.get_last_err_msg()}"
                    )
                logger.info(f"[MemFabricPool] no FAR contributor ready yet (ret={ret}), retrying ...")
                time.sleep(5)
                continue
            gva = int(info["gva"])
            contributor = int(info["rank_id"])
            self.allocator.add_block(gva, request)
            self.block_table.append({"gva": gva, "size": request, "contributor_rank": contributor})
            logger.info(
                f"[MemFabricPool] torch_rank={torch_rank} acquired {request >> 20}MiB block at "
                f"gva=0x{gva:x} from contributor_rank={contributor}"
            )
            remaining -= request

        self.handle = handle
        self.worker.start()
        self.inited = True

    def _log_rank_mapping(self, torch_rank):
        """Log the torch_rank <-> auto-assigned ralloc_rank mapping for ops/debugging."""
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                mapping = [None] * dist.get_world_size()
                dist.all_gather_object(mapping, (torch_rank, self.ralloc_rank))
                if dist.get_rank() == 0:
                    logger.info_rank0(f"[MemFabricPool] torch_rank <-> ralloc_rank mapping: {mapping}")
        except Exception:  # pylint: disable=broad-except
            pass

    def register_dev(self, ptr, nbytes):
        """Register device memory so device RDMA can access it directly (refcounted).

        Without registration the copy path silently degrades to a 128MB-bounce
        host staging route inside memfabric (correct but slow).
        """
        if self.register_mode != "per_tensor" or self.handle is None:
            return
        with self._reg_lock:
            entry = self._reg_entries.get(ptr)
            if entry is None:
                ret = self.handle.register(ptr, nbytes)
                if ret != 0:
                    raise RuntimeError(f"register device memory failed: {self.mf.get_last_err_msg()}")
                self._reg_entries[ptr] = [1, nbytes]
            else:
                entry[0] += 1

    def unregister_dev(self, ptr):
        if self.register_mode != "per_tensor" or self.handle is None:
            return
        with self._reg_lock:
            entry = self._reg_entries.get(ptr)
            if entry is None:
                return
            entry[0] -= 1
            if entry[0] <= 0:
                del self._reg_entries[ptr]
                ret = self.handle.unregister(ptr)
                if ret != 0:
                    logger.warning(f"unregister device memory failed at 0x{ptr:x}: {self.mf.get_last_err_msg()}")

    def copy_l2g(self, dev_ptr, gva, nbytes):
        """Copy local device (NPU) memory into the remote DRAM pool (LD2GH)."""
        if self.handle.copy_data(dev_ptr, gva, nbytes, 0) != 0:
            raise RuntimeError(f"copy_data L2G failed: {self.mf.get_last_err_msg()}")

    def copy_g2l(self, gva, dev_ptr, nbytes):
        """Copy from the remote DRAM pool back into local device memory (GH2LD)."""
        if self.handle.copy_data(gva, dev_ptr, nbytes, 0) != 0:
            raise RuntimeError(f"copy_data G2L failed: {self.mf.get_last_err_msg()}")

    def alloc_slot(self, nbytes):
        slot = self.allocator.alloc(nbytes)
        if slot is None:
            self._alloc_fail_count += 1
            if self._alloc_fail_count % 100 == 1:
                logger.warning(
                    f"[MemFabricPool] pool exhausted (used={self.allocator.used_size >> 20}MiB, "
                    f"total={self.allocator.total_size >> 20}MiB), skipping offload"
                )
        return slot

    def free_slot(self, gva):
        if gva is not None:
            self.allocator.free(gva)

    def destroy(self):
        if not self.inited:
            return
        self.worker.stop()
        try:
            self.handle.destroy()
            self.ralloc.uninitialize(0)
            self.mf.uninitialize()
        except Exception:  # pylint: disable=broad-except
            logger.warning("[MemFabricPool] destroy raised, ignoring")
        self.inited = False


class MFSwapTensor:
    """SwapTensor twin backed by the MemFabric remote DRAM pool.

    Interface-compatible with SwapTensor (launch_d2h/wait_d2h_finished/
    launch_h2d/prefetch_launch_h2d/wait_h2d_finished) so OffloadManager block
    bookkeeping and prefetch logic can be reused unchanged. NPU stream args are
    accepted but ignored: copies run through copy_data on the worker thread,
    coordinated via NPU events recorded on the compute stream.
    """

    def __init__(self, tensor, key, pool, register_mode="per_tensor", offload_enabled=True):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.key = key
        self.pool = pool
        self.register_mode = register_mode
        self.offload_enabled = offload_enabled and self.storage_size > 0
        self.nbytes = tensor.numel() * tensor.element_size()
        self.gva = None
        self.stat = "device"
        self._d2h_ptr = None
        self._d2h_task = None
        self._h2d_task = None

    # device to host: enqueue blocking copy onto the worker thread
    def launch_d2h(self, stream=None):
        if not self.offload_enabled or self.stat != "device":
            return
        gva = self.pool.alloc_slot(self.nbytes)
        if gva is None:
            self.offload_enabled = False  # degrade: keep activation on device
            return
        self.gva = gva
        self._d2h_ptr = self.tensor.data_ptr()
        compute_event = torch_npu.npu.Event()
        compute_event.record()
        pool, ptr, nbytes = self.pool, self._d2h_ptr, self.nbytes

        def _job():
            # wait until forward compute producing this tensor has finished
            compute_event.synchronize()
            if self.register_mode == "per_tensor":
                pool.register_dev(ptr, nbytes)
            pool.copy_l2g(ptr, gva, nbytes)

        self._d2h_task = pool.worker.submit(_job)

    # synchronize d2h, release device memory and the MR registration
    def wait_d2h_finished(self, stream=None, flag=False):
        if self._d2h_task is None:
            return
        self._d2h_task.wait()
        self._d2h_task = None
        if self.register_mode == "per_tensor":
            self.pool.unregister_dev(self._d2h_ptr)
        self._d2h_ptr = None
        self.tensor.storage().resize_(0)
        self.stat = "host"

    def _restore_storage(self):
        if self.tensor.storage().size() != self.storage_size:
            self.tensor.storage().resize_(self.storage_size)

    # host to device: synchronous copy on the calling (backward) thread
    def launch_h2d(self, h2d_stream=None, flag=True, working_stream=None):
        if self._h2d_task is not None:  # already prefetched, just wait it out
            self.wait_h2d_finished()
            return
        if self.stat != "host" or self.gva is None:
            return
        self._restore_storage()
        ptr = self.tensor.data_ptr()
        if self.register_mode == "per_tensor":
            self.pool.register_dev(ptr, self.nbytes)
        self.pool.copy_g2l(self.gva, ptr, self.nbytes)
        if self.register_mode == "per_tensor":
            self.pool.unregister_dev(ptr)
        self._release()
        self.stat = "device"

    # host to device ahead of time on the worker thread
    def prefetch_launch_h2d(self, h2d_stream=None, flag=True):
        if self.stat != "host" or self.gva is None or self._h2d_task is not None:
            return
        self._restore_storage()
        ptr = self.tensor.data_ptr()
        gva, nbytes, pool = self.gva, self.nbytes, self.pool

        def _job():
            if self.register_mode == "per_tensor":
                pool.register_dev(ptr, nbytes)
            pool.copy_g2l(gva, ptr, nbytes)
            if self.register_mode == "per_tensor":
                pool.unregister_dev(ptr)
            pool.free_slot(gva)

        self._h2d_task = pool.worker.submit(_job)
        self.stat = "device"

    def wait_h2d_finished(self):
        if self._h2d_task is not None:
            self._h2d_task.wait()
            self._h2d_task = None
            self.gva = None

    def _release(self):
        self.pool.free_slot(self.gva)
        self.gva = None


def async_save_on_cpu(
        h2d_stream,
        d2h_stream,
        block_idx,
        depth,
        custom_check_fn=None,
        prefetch=True,
        backend="pinned"
):
    """Context manager wrapping a transformer block for activation offload.

    Args:
        backend: "pinned" offloads to per-tensor pinned host buffers through NPU
            copy streams (original behaviour); "memfabric" offloads to the remote
            DRAM pool through MemFabric copy_data, executed on a worker thread
            and coordinated with NPU events (streams are unused in that case).
    """
    if backend == "memfabric":
        return _async_save_on_cpu_memfabric(block_idx, depth, custom_check_fn, prefetch)
    return _async_save_on_cpu_pinned(h2d_stream, d2h_stream, block_idx, depth, custom_check_fn, prefetch)


def _async_save_on_cpu_pinned(h2d_stream, d2h_stream, block_idx, depth, custom_check_fn=None, prefetch=True):
    class _PinnedHooks(saved_tensors_hooks):
        def __init__(self) -> None:

            def _pack_to_cpu(tensor):
                if not base_check_fn(tensor):
                    return tensor

                if (custom_check_fn is not None) and (not custom_check_fn(tensor)):
                    return tensor

                key, after_block = OffloadManager().get_cnt(block_idx)

                if after_block:
                    OffloadManager().del_npu_tensor("{}_".format(block_idx - 1), d2h_stream)

                swap_tensor = SwapTensor(tensor, key)

                if block_idx < depth - 1:
                    working_stream = torch_npu.npu.current_stream()
                    d2h_stream.wait_stream(working_stream)
                    swap_tensor.launch_d2h(d2h_stream)

                OffloadManager().put(key, swap_tensor)
                return swap_tensor

            def _unpack_from_cpu(swap_tensor) -> torch.Tensor:
                if isinstance(swap_tensor, torch.Tensor):
                    return swap_tensor

                working_stream = torch_npu.npu.current_stream()
                working_stream.wait_stream(h2d_stream)  # make sure all d2h copy is done before into backward

                h2d_stream.wait_stream(working_stream)

                swap_tensor.launch_h2d(h2d_stream, True, working_stream)

                if prefetch:
                    block_idx, tensor_idx = swap_tensor.key.split("_")
                    OffloadManager().prefetch_get(int(block_idx), int(tensor_idx), h2d_stream, d2h_stream)
                return swap_tensor.tensor

            super().__init__(_pack_to_cpu, _unpack_from_cpu)

    return _PinnedHooks()


def _async_save_on_cpu_memfabric(block_idx, depth, custom_check_fn=None, prefetch=True):
    class _MemFabricHooks(saved_tensors_hooks):
        def __init__(self) -> None:
            pool = MemFabricPool()

            def _pack_to_cpu(tensor):
                if not base_check_fn(tensor):
                    return tensor

                if (custom_check_fn is not None) and (not custom_check_fn(tensor)):
                    return tensor

                key, after_block = OffloadManager().get_cnt(block_idx)

                if after_block:
                    OffloadManager().del_npu_tensor("{}_".format(block_idx - 1), None)

                swap_tensor = MFSwapTensor(
                    tensor,
                    key,
                    pool,
                    register_mode=pool.register_mode,
                    offload_enabled=(block_idx < depth - 1),
                )

                if swap_tensor.offload_enabled:
                    swap_tensor.launch_d2h()

                OffloadManager().put(key, swap_tensor)
                return swap_tensor

            def _unpack_from_cpu(swap_tensor) -> torch.Tensor:
                if isinstance(swap_tensor, torch.Tensor):
                    return swap_tensor

                swap_tensor.launch_h2d()

                if prefetch:
                    blk_idx, tensor_idx = swap_tensor.key.split("_")
                    OffloadManager().prefetch_get(int(blk_idx), int(tensor_idx), None, None)
                return swap_tensor.tensor

            super().__init__(_pack_to_cpu, _unpack_from_cpu)

    return _MemFabricHooks()
