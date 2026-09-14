# Async Activation Offload

## 背景与挑战

随着大模型参数规模的增长和序列长度的上升，训练过程中对显存的需求急剧上升。目前对激活值显存的优化方案主要依赖于重计算技术和序列并行技术。这些技术存在以下瓶颈：

- 重计算通过丢弃前向传播中的激活值，并在反向传播时重新计算来节省显存，带来了大量的冗余计算。
- 序列并行虽然能将单个序列的计算分配到多个设备上达到降低显存的目的，但是频繁的跨设备通信可能难以被有效掩盖。

针对上述挑战，可以使用异步激活值卸载（Async Activation Offload）策略。

## 解决方案

- 显存优化：将激活值张量从device侧卸载至host侧，显著降低峰值显存占用。
- 异步执行：利用多流机制实现卸载（D2H）和加载（H2D）的异步，使拷贝过程被计算掩盖。
- 提前预取：反向过程中，通过`prefetch`机制提前加载后续需要的张量，隐藏加载延迟。

## 使用方法

该特性支持按“块”（block）组织张量生命周期，灵活管理模型不同block的激活值。使用示例如下：

```python
with async_save_on_cpu(
    h2d_stream=h2d_stream,
    d2h_stream=d2h_stream,
    block_idx=block_idx,
    depth=depth,
    custom_check_fn=your_check_fn
):
    # 模型某个block的前向计算代码，此处仅作为示例
    output = layer(input)
```

### 参数详解

- `h2d_stream`/`d2h_stream`：H2D和D2H流，建议全局单独新建一条流单独用来执行H2D和D2H任务，实现和计算流异步的效果。
- `block_idx`：当前block在模型中的编号。
- `depth`：模型的总层数。
- `custom_check_fn`：自定义校验函数，只有校验之后返回True的激活值才会被offload。建议根据实际情况筛选出计算量大、激活值参数量小的部分，并结合重计算策略：对于激活值参数量大、计算耗时短的进行重计算；对于激活值参数量小、计算耗时长的进行offload。否则H2D和D2H的开销过大，难以被计算掩盖。

## 使用案例及效果

- 长序列场景：self-attention的计算量随序列长度呈平方关系增长，使用该方案卸载self-attention前向计算的激活值，并在重计算时跳过self-attention的重计算。典型场景下端到端性能收益20%以上。
- FSDP2场景：FSDP2分布式策略下，对模型参数进行切分和聚合，较短序列长度下，计算耗时无法掩盖通信耗时。可以使用该方案，将重计算入口的激活值卸载，节省出显存后增大micro-batch size或序列长度提高计算比例。典型场景下端到端性能收益60%以上。

## MemFabric 远端 DRAM 内存池后端（FSDP2 / Qwen3-MoE）

除默认的 pinned host buffer 后端外，FSDP2 后端（当前接入 Qwen3-MoE）支持将激活值卸载到
[MemFabric](https://gitcode.com/Ascend/memfabric_hybrid) 跨机 DRAM 内存池，走 DEVICE_RDMA
传输，适用于本机 Host 内存不足或希望利用远端内存节点 DRAM 的场景。

### 部署时序（config store 托管在 FAR 侧，auto_rank 模式）

1. 在第一个远端内存节点启动 FAR 守护进程（托管 config store）：
   ```bash
   python examples/fsdp2/qwen3_moe/memfabric_far_daemon.py \
       --store-url tcp://<far_ip>:8572 --with-store \
       --nic tcp://<far_ip>:10005 --world-size 16
   ```
2. 在其余远端内存节点启动其他 FAR 守护进程（等待 store 后自动注册）。
3. 启动训练（NEAR 侧）。各训练 rank 以 `auto_ranking` 模式初始化 ralloc，记录
   `torch_rank <-> ralloc_rank` 映射与"已申请块 → 贡献者 rank"映射表；训练框架自身的
   rankId 不用于配置 ralloc。

### 训练侧参数（parallel 组）

| 参数 | 说明 |
|---|---|
| `activation_offload` | 开启激活值异步卸载 |
| `activation_offload_backend` | `memfabric`（默认）或 `pinned` |
| `offload_pool_size_gb` | 每 rank 向池申请的远端 DRAM 总量 |
| `offload_extend_block_gb` | 每次 `extend_remote_mem` 的申请粒度（2M 对齐） |
| `offload_register_mode` | `per_tensor`（默认）：拷贝前后注册/注销 NPU 张量内存，走一跳 DEVICE_RDMA 直写；`none`：不注册，走内部 128MB bounce 路径（较慢） |
| `mf_store_url` | FAR 侧托管的 config store 地址（必填） |
| `mf_nic` | DEVICE_RDMA 所需的 RoCE NIC 地址（必填） |
| `mf_world_size` | ralloc 窗口容量，需与 FAR 守护进程一致 |

完整示例参见 `examples/fsdp2/qwen3_moe/tune_qwen3_30b_4k_fsdp2_A2_memfabric.sh/.yaml`。

### 实现要点

- DEVICE_RDMA 的 `copy_data` 为 host 同步调用（不支持 async flag），异步性由后台卸载线程
  + NPU event 协调实现：pack 时记录计算流 event，线程内等 event 后发起 L2GH 拷贝，下个
  block 边界释放显存；反向 unpack 同步 GH2L 拷回并预取下一 block。
- 训练侧对远端池块做 4K 对齐细粒度子分配（块申请 2M 对齐），slot 随张量生命周期回收复用。
- 池耗尽时自动退化为"不卸载"，训练不中断。
