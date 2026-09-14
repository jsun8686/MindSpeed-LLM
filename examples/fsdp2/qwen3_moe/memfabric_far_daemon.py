# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""MemFabric FAR contributor daemon for activation offload.

Runs on every remote-memory node (one process per node). It registers itself as
a FAR-role placement candidate; when a NEAR training rank calls
extend_remote_mem, this daemon contributes its node's DRAM and joins the pool
on demand.

Startup order (store is hosted on the FAR side):
  1. Start the first FAR daemon with --with-store (it hosts the config store).
  2. Start the remaining FAR daemons (they wait for the store, then register).
  3. Start the NEAR training job (each rank waits for the store, then
     ralloc.initialize + ralloc.create + extend_remote_mem).

Usage:
  # first FAR node (hosts the store)
  python memfabric_far_daemon.py --store-url tcp://10.0.0.1:8572 --with-store \
      --nic tcp://10.0.0.1:10005 --world-size 64
  # other FAR nodes
  python memfabric_far_daemon.py --store-url tcp://10.0.0.1:8572 \
      --nic tcp://10.0.1.1:10005 --world-size 64
"""

import argparse
import signal
import socket
import sys
import time


def wait_for_store(store_url, timeout_sec, poll_interval=0.5):
    host, port = store_url.split("://", 1)[-1].rsplit(":", 1)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex((host, int(port))) == 0:
                return
        time.sleep(poll_interval)
    raise RuntimeError(f"config store not reachable within {timeout_sec}s: {store_url}")


def main():
    parser = argparse.ArgumentParser(description="MemFabric FAR contributor daemon")
    parser.add_argument("--store-url", required=True, help="config store url, e.g. tcp://ip:port")
    parser.add_argument(
        "--with-store",
        action="store_true",
        help="host the config store (set on exactly one FAR daemon, the one started first)",
    )
    parser.add_argument("--nic", required=True, help="RoCE nic url for RDMA transport, e.g. tcp://ip:port")
    parser.add_argument(
        "--world-size",
        type=int,
        default=64,
        help="ralloc window capacity; must match the training-side mf_world_size",
    )
    parser.add_argument("--timeout", type=int, default=300, help="store wait timeout in seconds")
    parser.add_argument("--device-id", type=int, default=0, help="local device id")
    args = parser.parse_args()

    import memfabric_hybrid as mf
    from memfabric_hybrid import ralloc

    mf.set_log_level(1)
    assert mf.initialize() == 0, "mf.initialize failed"

    ralloc_inited = False
    running = True

    def _stop(signum, frame):  # pylint: disable=unused-argument
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        cfg = ralloc.RallocConfig()
        cfg.role = ralloc.RallocRole.FAR  # resident contributor node
        cfg.auto_ranking = True  # rank ids auto-assigned by ralloc
        cfg.start_store = args.with_store  # the store is hosted on the FAR side
        cfg.set_nic(args.nic)

        if not args.with_store:
            wait_for_store(args.store_url, args.timeout)

        assert ralloc.initialize(args.store_url, args.world_size, args.device_id, cfg) == 0, (
            f"ralloc.initialize failed: {mf.get_last_err_msg()}"
        )
        ralloc_inited = True

        rank_id = ralloc.get_rank_id()
        print(f"[far-daemon] registered as FAR contributor, ralloc_rank={rank_id}, store={args.store_url}", flush=True)
        print("[far-daemon] contributing DRAM to NEAR pools on demand, press Ctrl+C to exit", flush=True)

        while running:
            time.sleep(1)

        print("[far-daemon] shutting down", flush=True)
    finally:
        if ralloc_inited:
            ralloc.uninitialize(0)
        mf.uninitialize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
