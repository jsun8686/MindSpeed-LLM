"""Async activation offload feature (MemFabric / pinned backends) for the mcore path.

Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ActivationOffloadFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__(feature_name='activation-offload', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--activation-offload', action='store_true', default=False,
                           help='Offload per-block layer-input activations asynchronously '
                                '(saved_tensors_hooks based, works for all GPT-style models).')
        group.add_argument('--activation-offload-backend', type=str, default='memfabric',
                           choices=['pinned', 'memfabric'],
                           help="Backend used for activation offload: 'pinned' copies to per-tensor "
                                "pinned host buffers through NPU streams; 'memfabric' offloads to the "
                                "remote DRAM memory pool through MemFabric ralloc (DEVICE_RDMA).")
        group.add_argument('--offload-pool-size-gb', type=int, default=8,
                           help='Total remote DRAM (GiB) acquired per rank from the MemFabric pool.')
        group.add_argument('--offload-extend-block-gb', type=int, default=1,
                           help='Granularity (GiB) of each extend_remote_mem request (2M aligned).')
        group.add_argument('--offload-register-mode', type=str, default='per_tensor',
                           choices=['per_tensor', 'none'],
                           help="per_tensor: register device memory around each copy (one-hop device "
                                "RDMA); none: internal 128MB bounce path (slower).")
        group.add_argument('--mf-store-url', type=str, default=None,
                           help='FAR-hosted ralloc config store url (tcp://ip:port), required for '
                                'the memfabric backend.')
        group.add_argument('--mf-nic', type=str, default=None,
                           help='RoCE nic url required by device RDMA, required for the memfabric backend.')
        group.add_argument('--mf-world-size', type=int, default=64,
                           help='ralloc window capacity; must match the FAR daemons.')
        group.add_argument('--mf-pool-id', type=int, default=0,
                           help='ralloc pool id, in [0, 63].')
        group.add_argument('--mf-store-wait-timeout', type=int, default=300,
                           help='Seconds to wait for the FAR-hosted store / contributors.')

    def validate_args(self, args):
        if not getattr(args, 'activation_offload', False):
            return

        if args.activation_offload_backend == 'memfabric':
            if not args.mf_store_url:
                raise AssertionError('--mf-store-url is required when --activation-offload-backend '
                                     'is memfabric.')
            if not args.mf_nic:
                raise AssertionError('--mf-nic is required when --activation-offload-backend '
                                     'is memfabric.')
        if args.offload_pool_size_gb <= 0:
            raise AssertionError('--offload-pool-size-gb must be positive.')
        if args.offload_extend_block_gb <= 0 or args.offload_extend_block_gb > args.offload_pool_size_gb:
            raise AssertionError('--offload-extend-block-gb must be in (0, offload-pool-size-gb].')

        # v1 scope guards: TP/DP(+EP) only.
        if args.pipeline_model_parallel_size > 1:
            raise AssertionError('--activation-offload currently supports pipeline-parallel-size == 1.')
        if getattr(args, 'num_layers_per_virtual_pipeline_stage', None):
            raise AssertionError('--activation-offload does not support virtual pipeline (VPP) yet.')
        if getattr(args, 'context_parallel_size', 1) > 1:
            raise AssertionError('--activation-offload does not support context-parallel yet (unverified).')
        if getattr(args, 'share_kvstates', False):
            raise AssertionError('--activation-offload does not support share-kvstates (GDN) models yet.')
        if getattr(args, 'n_hash_layers', 0) and args.n_hash_layers >= 1:
            raise AssertionError('--activation-offload does not support n-hash-layers models yet.')
        if getattr(args, 'recompute_granularity', None) == 'full':
            raise AssertionError('--activation-offload with full recompute-granularity is not verified '
                                 'yet (hook nesting with mcore checkpointing); use selective/none for now.')
