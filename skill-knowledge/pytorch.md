# PyTorch 核心特性 — 代码级深度分析

> 本文档聚焦 PyTorch 提供的**训练基础设施**——上层所有训练仓（Megatron/torchtitan/DeepSpeed/miles/slime）的底层依赖。每个技术主题包含：概念原理 + 完整调用链（带 file:line）+ ASCII 架构图 + 关键类/函数表 + 配置参数表。

---

## 0. PyTorch 训练栈全景

PyTorch 训练栈由六个核心层级构成，自底向上依次为：CUDA 运行时 → 分布式通信 → 自动微分 → 神经网络模块 → 优化器 → 编译后端。上层训练框架（Megatron-LM、DeepSpeed、torchtitan）在这些基础组件之上构建自己的并行策略和调度逻辑。

### 1.1 训练栈层级架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     上层训练框架                                      │
│   Megatron-LM   DeepSpeed   torchtitan   miles   slime             │
├─────────────────────────────────────────────────────────────────────┤
│  torch.compile (Inductor)  │  Distributed Checkpoint (DCP)          │
├─────────────────────────────────────────────────────────────────────┤
│  torch.optim (AdamW, SGD)  │  lr_scheduler (Cosine, Linear)        │
├─────────────────────────────────────────────────────────────────────┤
│  torch.nn (Module, Linear, LayerNorm, Embedding, FSDP, DTensor)     │
├─────────────────────────────────────────────────────────────────────┤
│  torch.autograd (backward, Function, grad_mode)                     │
├─────────────────────────────────────────────────────────────────────┤
│  torch.distributed (ProcessGroup, all_reduce, all_gather, FSDP2)    │
├─────────────────────────────────────────────────────────────────────┤
│  torch.cuda (CUDAGraph, CachingAllocator, Stream, Event, NCCL)     │
├─────────────────────────────────────────────────────────────────────┤
│  CUDA Runtime / NCCL / HCCL (硬件通信层)                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心模块表

| 模块 | 核心类/函数 | 训练作用 | 源码位置 |
|------|-----------|---------|---------|
| `torch.nn` | `Module`, `Parameter`, `Linear` | 模型定义、参数注册 | `torch/nn/modules/module.py` |
| `torch.autograd` | `backward`, `Function`, `no_grad` | 自动微分引擎 | `torch/autograd/__init__.py` |
| `torch.optim` | `Optimizer`, `AdamW`, `LRScheduler` | 参数更新 | `torch/optim/optimizer.py` |
| `torch.distributed` | `ProcessGroup`, `all_reduce`, `broadcast` | 集合通信 | `torch/distributed/distributed_c10d.py` |
| `torch.distributed.fsdp` | `FullyShardedDataParallel`, `FlatParameter` | 参数分片并行 | `torch/distributed/fsdp/fully_sharded_data_parallel.py` |
| `torch.distributed.tensor` | `DTensor`, `DeviceMesh`, `Shard` | 分布式张量抽象 | `torch/distributed/tensor/_api.py` |
| `torch.distributed.pipelining` | `PipelineStage`, `Schedule1F1B` | 流水线并行 | `torch/distributed/pipelining/stage.py` |
| `torch.distributed.checkpoint` | `get_state_dict`, `set_state_dict` | 分布式 checkpoint | `torch/distributed/checkpoint/state_dict.py` |
| `torch.cuda` | `CUDAGraph`, `MemPool`, `empty_cache` | CUDA 图、内存管理 | `torch/cuda/graphs.py` |
| `torch._inductor` | `compile_fx`, `compile_fx_inner` | torch.compile 后端 | `torch/_inductor/compile_fx.py` |
| `torch.multiprocessing` | `spawn`, `start_processes` | 多进程启动 | `torch/multiprocessing/__init__.py` |

---

## 1. torch.nn — 神经网络基础

`torch.nn` 是 PyTorch 的神经网络基础层，核心抽象是 `nn.Module`。所有模型都继承自 `nn.Module`，通过注册参数（`Parameter`）、缓冲区（`Buffer`）和子模块（`Module`）来构建层级结构。

### 2.1 nn.Module 核心机制

**概念原理**：`nn.Module` 通过 `__setattr__` 魔术方法自动识别赋值类型——`Parameter` 被注册到 `_parameters` 字典，`Module` 被注册到 `_modules` 字典，普通 `Tensor` 可通过 `register_buffer` 注册到 `_buffers` 字典。`__call__` 触发 `_call_impl`，按顺序执行 forward pre-hooks → forward → forward hooks，并设置 backward hooks。

#### nn.Module.__init__ 初始化

`Module.__init__` 初始化所有内部状态字典（`module.py:482`）：

- `_parameters: dict[str, Parameter | None]` — 存储可学习参数
- `_buffers: dict[str, Tensor | None]` — 存储不参与梯度的持久状态
- `_modules: dict[str, Optional[Module]]` — 存储子模块
- `_forward_hooks`, `_forward_pre_hooks`, `_backward_hooks`, `_backward_pre_hooks` — 钩子字典
- `_state_dict_hooks`, `_load_state_dict_pre_hooks`, `_load_state_dict_post_hooks` — state_dict 钩子

#### nn.Module.__call__ 调用链

```
model(input)
  → Module.__call__                          # module.py:1919
    → Module._wrapped_call_impl              # module.py:1776
      → [if compiled] _compiled_call_impl    # torch.compile 路径
      → [else] Module._call_impl             # module.py:1784
        → forward_pre_hooks (global + local) # module.py:1807-1827
        → BackwardHook.setup_input_hook      # module.py:1830-1832
        → forward_call(*args, **kwargs)      # module.py:1834
          → [tracing] Module._slow_forward   # module.py:1756
          → [eager]  Module.forward
        → forward_hooks (global + local)     # module.py:1835-1850
        → BackwardHook.setup_output_hook     # module.py:1857
        → non-full backward hooks register   # module.py:1860-1871
```

#### nn.Module.apply / modules() / children()

```python
# torch/nn/modules/module.py:1076
def apply(self, fn):
    for module in self.children():
        module.apply(fn)
    fn(self)
    return self

# torch/nn/modules/module.py:2782
def children(self):
    for _name, module in self.named_children():
        yield module

# torch/nn/modules/module.py:2811
def modules(self, remove_duplicate=True):
    for _, module in self.named_modules(remove_duplicate=remove_duplicate):
        yield module
```

#### nn.Module.parameters() / named_parameters()

```python
# torch/nn/modules/module.py:2667
def parameters(self, recurse=True):
    for _name, param in self.named_parameters(recurse=recurse):
        yield param

# torch/nn/modules/module.py:2696
def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
    gen = self._named_members(
        lambda module: module._parameters.items(),
        prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate,
    )
    yield from gen
```

#### nn.Module.state_dict() / load_state_dict()

```python
# torch/nn/modules/module.py:2196
def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
    # ... 初始化 destination OrderedDict
    for hook in self._state_dict_pre_hooks.values():
        hook(self, prefix, keep_vars)
    self._save_to_state_dict(destination, prefix, keep_vars)  # module.py:2145
    for name, module in self._modules.items():
        if module is not None:
            module.state_dict(destination=destination,
                              prefix=prefix + name + ".", keep_vars=keep_vars)
    for hook in self._state_dict_hooks.values():
        hook_result = hook(self, destination, prefix, local_metadata)
    return destination

# torch/nn/modules/module.py:2532
def load_state_dict(self, state_dict, strict=True, assign=False):
    def load(module, local_state_dict, prefix=""):
        module._load_from_state_dict(
            local_state_dict, prefix, local_metadata, True,
            missing_keys, unexpected_keys, error_msgs)  # module.py:2347
        for name, child in module._modules.items():
            if child is not None:
                load(child, child_state_dict, child_prefix)
    load(self, state_dict)
    # strict 模式下检查 missing/unexpected keys
```

### 2.2 Parameter 与 Buffer

**概念原理**：`Parameter` 是 `Tensor` 的子类，通过 `_is_param` 标志位和 `_ParameterMeta` 元类实现。当 `Parameter` 被赋值给 `Module` 属性时，`__setattr__` 自动调用 `register_parameter`。`Buffer` 通过 `register_buffer` 注册，不参与梯度计算，但会随模型移动设备。

```python
# torch/nn/parameter.py:30
class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            return torch.Tensor._make_subclass(cls, data, requires_grad)
        t = data.detach().requires_grad_(requires_grad)
        t._is_param = True
        return t
```

| 类型 | requires_grad | 参与优化 | state_dict | 典型用途 |
|------|--------------|---------|------------|---------|
| `Parameter` | True（默认） | 是 | 是 | 权重、偏置 |
| `Buffer` | False | 否 | 是（persistent=True） | BatchNorm running_mean |
| 普通 Tensor | 取决于设置 | 否 | 否 | 临时计算结果 |

### 2.3 关键模块

```python
# torch/nn/modules/linear.py:53
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        self.reset_parameters()  # Kaiming uniform 初始化

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
```

### 2.4 nn.Module 调用链 ASCII 图

```
                    ┌──────────────────────┐
                    │   model = MyModel()  │
                    └──────────┬───────────┘
                               │ __init__
                               ▼
                    ┌──────────────────────┐
                    │  _parameters = {}    │
                    │  _buffers = {}       │
                    │  _modules = {}       │
                    └──────────┬───────────┘
                               │ model(input)
                               ▼
                    ┌──────────────────────┐
                    │  Module.__call__     │  :1919
                    │  → _wrapped_call_impl│  :1776
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │forward_pre   │ │  _call_impl  │ │ backward     │
     │_hooks        │ │  :1784       │ │ hooks setup  │
     └──────────────┘ └──────┬───────┘ └──────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │  forward_call()      │
                    │  → _slow_forward     │  :1756
                    │  → Module.forward    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  forward_hooks       │
                    │  setup_output_hook   │
                    └──────────────────────┘
```

---

## 2. torch.autograd — 自动微分

`torch.autograd` 是 PyTorch 的自动微分引擎，通过动态计算图（Dynamic Computation Graph）实现反向传播。每个 `Tensor` 的 `grad_fn` 指向创建它的 `Function` 节点，形成链式结构。

### 3.1 backward() 调用链

**概念原理**：`backward()` 从输出张量出发，沿 `grad_fn` 链式调用每个 `Function` 的 `backward` 方法，将梯度累积到叶子节点的 `.grad` 属性。支持 `retain_graph`（保留计算图用于多次反向传播）和 `create_graph`（构建微分图以计算高阶导数）。

```python
# torch/autograd/__init__.py:255
def backward(tensors, grad_tensors=None, retain_graph=None,
             create_graph=False, grad_variables=None, inputs=None):
    # ... 参数校验、functorch 检查
    grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
    if retain_graph is None:
        retain_graph = create_graph
    _engine_run_backward(
        tensors, grad_tensors_, retain_graph, create_graph,
        inputs_tuple, allow_unreachable=True, accumulate_grad=True,
    )
```

#### backward 完整调用链

```
loss.backward()
  → autograd.backward(tensors=[loss])           # __init__.py:255
    → _make_grads(tensors, grad_tensors)         # __init__.py:91
      → 检查 shape/dtype 匹配                    # __init__.py:130-195
      → 标量输出自动创建 ones_like grad           # __init__.py:230-234
    → _engine_run_backward(                      # __init__.py:395
        tensors, grad_tensors_, retain_graph,
        create_graph, inputs, accumulate_grad=True
      )
      → C++ autograd engine 遍历 grad_fn 链
        → 每个 Node 调用 registered hooks
        → 到达叶子节点 → accumulate_grad 到 .grad
```

### 3.2 Function 自定义 autograd

**概念原理**：通过继承 `torch.autograd.Function` 并实现 `forward` / `backward` 静态方法，可以定义自定义的前向/反向计算逻辑。`ctx` 对象用于在 forward 和 backward 之间传递数据。

```python
# torch/autograd/function.py:364
class _SingleLevelFunction(_C._FunctionBase, FunctionCtx, _HookMixin,
                           metaclass=FunctionMeta):
    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError(
            "You must implement the forward function for custom autograd.Function.")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError(
            "You must implement either the backward or vjp method...")

    vjp = backward  # vjp 和 backward 是别名
```

| 方法 | 作用 | 调用时机 |
|------|------|---------|
| `forward(ctx, *args)` | 前向计算 | 张量运算时自动调用 |
| `backward(ctx, *grad_outputs)` | 反向梯度计算 | `loss.backward()` 时链式调用 |
| `ctx.save_for_backward(*tensors)` | 保存中间张量供 backward 使用 | forward 中调用 |
| `ctx.save_for_forward(*tensors)` | 保存中间张量供 jvp 使用 | forward 中调用 |

### 3.3 no_grad / enable_grad 上下文

`torch.autograd.grad_mode` 提供梯度模式控制：

- `no_grad()` — 禁用梯度计算，减少内存占用
- `enable_grad()` — 启用梯度计算
- `set_grad_enabled(mode)` — 条件切换梯度模式
- `inference_mode()` — 比 no_grad 更激进的性能优化

---

## 3. torch.optim — 优化器

`torch.optim` 提供参数更新的数学实现。所有优化器继承自 `Optimizer` 基类，通过 `param_groups` 管理不同参数组的超参数，通过 `state` 维护每个参数的优化器状态（如动量、方差）。

### 4.1 Optimizer 基类

**概念原理**：`Optimizer` 将参数组织为 `param_groups`（列表，每个元素是包含 `params` 和如 `lr`、`weight_decay` 等超参数的字典）。`state` 是一个 `defaultdict(dict)`，以参数 ID 为键存储优化器状态。

```python
# torch/optim/optimizer.py:339
class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)         # 参数状态（动量、方差等）
        self.param_groups = []                 # 参数组列表
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)
```

#### Optimizer.step() / zero_grad()

```python
# torch/optim/optimizer.py:1093
def step(self, closure=None):
    raise NotImplementedError  # 子类实现

# torch/optim/optimizer.py:1024
def zero_grad(self, set_to_none=True):
    for group in self.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    if not foreach or p.grad.is_sparse:
                        p.grad.zero_()
                    else:
                        per_device_and_dtype_grads[
                            p.grad.device][p.grad.dtype].append(p.grad)
    if foreach:
        for per_dtype_grads in per_device_and_dtype_grads.values():
            for grads in per_dtype_grads.values():
                torch._foreach_zero_(grads)
```

#### Optimizer 调用链

```
optimizer.step()
  → Optimizer.profile_hook_step(step)         # optimizer.py:508
    → global pre_hooks + local pre_hooks      # optimizer.py:516-527
    → func(*args, **kwargs)                   # 实际 step 实现
      → [AdamW] Adam.step → _single_tensor_adamw 或 _foreach_adamw
    → _optimizer_step_code()                  # profiler hook
    → local post_hooks + global post_hooks    # optimizer.py:534-538
```

### 4.2 AdamW 实现

**概念原理**：AdamW 是 Adam 的解耦权重衰减变体。与 L2 正则化不同，AdamW 在参数更新前直接将权重衰减应用于参数（`θ_t ← θ_{t-1} - γλθ_{t-1}`），而不是混入梯度。

```python
# torch/optim/adamw.py:20
class AdamW(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize=False,
                 foreach=None, capturable=False, differentiable=False,
                 fused=None):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad,
                         foreach=foreach, maximize=maximize,
                         capturable=capturable, differentiable=differentiable,
                         fused=fused,
                         decoupled_weight_decay=True)  # 关键区别
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `lr` | 1e-3 | 学习率 |
| `betas` | (0.9, 0.999) | 一阶/二阶矩衰减率 |
| `eps` | 1e-8 | 数值稳定项 |
| `weight_decay` | 1e-2 | 解耦权重衰减系数 |
| `amsgrad` | False | 是否使用 AMSGrad 变体 |
| `foreach` | None | 是否使用 foreach 批量实现 |
| `fused` | None | 是否使用 fused CUDA 实现 |
| `capturable` | False | 是否可在 CUDA Graph 中捕获 |

### 4.3 lr_scheduler

`torch.optim.lr_scheduler` 提供学习率调度策略，常见包括：

- `LinearLR` — 线性 warmup
- `CosineAnnealingLR` — 余弦退火
- `SequentialLR` — 组合多个 scheduler
- `StepLR` / `MultiStepLR` — 阶梯衰减

---

## 4. torch.distributed — 分布式通信

`torch.distributed` 提供跨进程/跨节点的集合通信原语。核心抽象是 `ProcessGroup`（进程组）和 `Backend`（通信后端），上层构建 `all_reduce`、`broadcast`、`all_gather`、`reduce_scatter` 等集合操作。

### 5.1 进程组初始化

**概念原理**：`init_process_group` 是分布式训练的入口函数，它通过 rendezvous 机制协调所有进程，创建默认的 `ProcessGroup` 并注册到全局 `_World` 单例。支持多种后端：NCCL（GPU）、Gloo（CPU）、MPI、HCCL（Ascend NPU）。

```python
# torch/distributed/distributed_c10d.py:2350+
def init_process_group(backend="undefined", init_method=None, timeout=None,
                       world_size=-1, rank=-1, store=None, group_name="",
                       pg_options=None, device_id=None):
    # ... backend 解析、rendezvous
    default_pg, _ = _new_process_group_helper(
        world_size, rank, [], backend, store, group_name,
        backend_options=pg_options, timeout=timeout,
        device_id=device_id, group_desc="default_pg",
    )
    _update_default_pg(default_pg)
    # ... barrier、excepthook 设置
```

#### init_process_group 调用链

```
torch.distributed.init_process_group(backend="nccl")
  → Backend(backend) 解析                     # distributed_c10d.py:2338
  → [非 MPI] rendezvous(init_method, rank, world_size)  # :2383
    → store, rank, world_size = next(rendezvous_iterator)
    → store = PrefixStore("default_pg", store)  # :2391
  → _new_process_group_helper(...)              # :2393
    → _create_nccl_process_group(opts, backend_options)  # :590
      → ProcessGroupNCCL(store, rank, size, timeout, options)
    → pg._register_backend(device, backend_type, backend_class)  # :2762
    → _register_pg_in_world(pg, ...)           # :2784
  → _update_default_pg(default_pg)              # :2407
  → [可选] barrier() 或 _store_based_barrier()   # :2435-2454
```

### 5.2 集合通信原语

**概念原理**：所有集合通信原语最终通过 `ProcessGroup` 的 C++ 实现完成。NCCL 后端使用 `ProcessGroupNCCL`，Gloo 后端使用 `ProcessGroupGloo`。Python 层负责参数校验、`__torch_function__` 分发和异常处理。

```python
# torch/distributed/distributed_c10d.py:3700+
@_exception_logger
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    # ... __torch_function__ 检查
    group = _group_or_default_group(group)
    opts = AllreduceOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op
    work = group.allreduce([tensor], opts)
    if async_op:
        return work
    elif work is not None:
        work.wait()

# torch/distributed/distributed_c10d.py:3548
@_exception_logger
def broadcast(tensor, src=None, group=None, async_op=False, group_src=None):
    group = _group_or_default_group(group)
    opts = BroadcastOptions()
    opts.rootRank = group_src
    work = group.broadcast([tensor], opts)
```

| 原语 | 作用 | 典型用途 |
|------|------|---------|
| `all_reduce` | 全归约，所有进程得到相同结果 | 梯度同步 |
| `broadcast` | 广播，一个进程发送数据到所有进程 | 参数分发 |
| `all_gather` | 全收集，拼接所有进程的数据 | FSDP 参数重建 |
| `reduce_scatter` | 归约+散射，每个进程得到一部分结果 | FSDP 梯度同步 |
| `barrier` | 同步屏障 | 阶段同步 |

### 5.3 通信调用链

```
dist.all_reduce(tensor)
  → all_reduce(tensor, op=SUM, group=None)    # distributed_c10d.py:3700
    → _exception_logger 包装
    → group = _group_or_default_group(group)    # 获取 ProcessGroup
    → opts = AllreduceOptions()
    → work = group.allreduce([tensor], opts)    # C++ ProcessGroupNCCL.allreduce
      → ncclAllReduce(input, output, count, dtype, op, comm, stream)
    → [async_op] return work
    → [sync] work.wait()
```

---

## 5. FSDP2（Fully Sharded Data Parallel）

FSDP2 是 PyTorch 2.x 推出的新一代数据并行方案，相比 FSDP1（`FullyShardedDataParallel`）更轻量、更灵活。核心思想是将模型参数按 rank 分片（shard），前向/反向时按需 all-gather 重建完整参数，反向结束后通过 reduce-scatter 同步梯度并释放全量参数。

### 6.1 FSDP2 核心机制

**概念原理**：FSDP2 通过 `fully_shard()` 函数（而非 FSDP1 的包装类）对模块参数进行分片。每个参数被包装为 `DTensor`，使用 `Shard(0)` 放置策略。FSDP2 使用 `FSDPParamGroup` 管理参数组，通过 `all_gather_into_tensor` 重建参数，通过 `reduce_scatter_tensor` 同步梯度。

```python
# torch/distributed/_composable/fsdp/_fsdp_init.py
# fully_shard() 是 FSDP2 的入口函数
def fully_shard(module, *, mesh, mp_policy, reshard_after_forward, ...):
    # 将模块参数转换为 DTensor 并分片
    ...
```

#### FSDP2 参数生命周期

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FSDP2 参数生命周期                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐   all-gather   ┌─────────────┐   forward/backward  │
│  │ Sharded     │ ─────────────→ │ Unsharded   │ ──────────────────→ │
│  │ Param       │                │ Param       │                     │
│  │ (1/N 大小)  │ ←───────────── │ (完整大小)  │                     │
│  └─────────────┘   re-shard     └─────────────┘                     │
│        ↑                                        │                   │
│        │         reduce-scatter                  │                   │
│        └────────────────────────────────────────┘                   │
│                 梯度同步 + 参数释放                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 FSDP1 vs FSDP2 对比

| 特性 | FSDP1 (`FullyShardedDataParallel`) | FSDP2 (`fully_shard`) |
|------|-----------------------------------|----------------------|
| API 风格 | 包装类（Module Wrapper） | 函数式 composable API |
| 参数管理 | `FlatParamHandle` / `FlatParameter` | 原生 `DTensor` + `FSDPParamGroup` |
| 分片粒度 | 模块级（整个模块参数 flatten） | 参数级（per-parameter） |
| 通信优化 |  prefetch、overlap | 更灵活的 mesh 控制 |
| 状态管理 | `_get_state_dict_type` / `StateDictType` | 与 DCP 集成 |
| 源码位置 | `fsdp/fully_sharded_data_parallel.py:397` | `_composable/fsdp/` |

### 6.3 FSDP1 核心实现（遗留系统参考）

```python
# torch/distributed/fsdp/fully_sharded_data_parallel.py:397
class FullyShardedDataParallel(Module, _FSDPState):
    def __init__(self, module, process_group=None, sharding_strategy=None,
                 cpu_offload=None, auto_wrap_policy=None, ...):
        super().__init__()
        self._fsdp_wrapped_module = module
        # ... 初始化 FlatParamHandle

# torch/distributed/fsdp/_flat_param.py:481
class FlatParamHandle:
    def __init__(self, params, module, device, ...):
        # 将多个参数 flatten 为单个 FlatParameter
        self.fsdp_params = [FlatParameter(params, requires_grad=True)]
```

### 6.4 FSDP2 vs 上层框架对比

| 框架 | 数据并行方案 | 分片粒度 | 通信原语 | 特点 |
|------|------------|---------|---------|------|
| PyTorch FSDP2 | `fully_shard` + DTensor | 参数级 | all-gather / reduce-scatter | 原生 composable |
| Megatron DistributedOptimizer | DistributedOptimizer | 参数级（连续分片） | reduce-scatter / all-gather | 与 TP 深度集成 |
| DeepSpeed ZeRO-1/2/3 | ZeROStage | 参数/梯度/状态级 | all-gather / reduce-scatter | 三阶段渐进分片 |
| DeepSpeed-0 | ZeRO-0 | 无分片 | all_reduce | 传统 DDP |

---

## 6. DTensor 与 Tensor Parallel

`torch.distributed.tensor`（DTensor）是 PyTorch 2.x 引入的分布式张量抽象，为 Tensor Parallel（TP）和 Sequence Parallel 提供统一的设备无关描述。

### 7.1 DTensor 核心概念

**概念原理**：`DTensor` 由三部分描述：`local_tensor`（本地物理张量）、`DeviceMesh`（设备拓扑）、`Placement`（每个维度的分布方式）。Placement 有三种：`Shard(dim)`（按维度切分）、`Replicate`（全量复制）、`Partial`（待归约的部分和）。

```python
# torch/distributed/tensor/_api.py:356
class DTensor(torch.Tensor, metaclass=_FromTorchTensor):
    def __new__(cls, local_tensor, device_mesh, placements, ...):
        ...

# torch/distributed/tensor/placement_types.py
class Placement: pass
class Shard(Placement):    # 按维度切分
    def __init__(self, dim: int): self.dim = dim
class Replicate(Placement): pass  # 全量复制
class Partial(Placement):         # 部分和（待归约）
    def __init__(self, reduce_op="sum"): ...
```

#### DeviceMesh 初始化

```python
# torch/distributed/device_mesh.py:1544
def init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None):
    # 创建设备网格，如 init_device_mesh("cuda", (8,), ["tp"])
    # 返回 DeviceMesh 对象
    ...
```

### 7.2 Tensor Parallel 实现

**概念原理**：PyTorch TP 通过 `parallelize_module` 将 `ColwiseParallel` 和 `RowwiseParallel` 应用于线性层。`ColwiseParallel` 将权重按列切分，输入复制到所有 rank，输出为 `Partial`（需要 all-reduce）。`RowwiseParallel` 将权重按行切分，输入为 `Shard`，输出为 `Partial`。

```python
# torch/distributed/tensor/parallel/api.py:14
def parallelize_module(module, mesh, parallelize_plan):
    # 遍历 plan 中的 ParallelStyle，应用到模块
    ...

# torch/distributed/tensor/parallel/style.py:45
class ColwiseParallel(ParallelStyle):
    # 列切分：weight 按列分，输入 Replicate，输出 Partial
    ...

class RowwiseParallel(ParallelStyle):
    # 行切分：weight 按行分，输入 Shard，输出 Partial
    ...
```

#### TP 矩阵乘法示意

```
ColwiseParallel Linear:
┌──────────────────────────────────────────────────────┐
│  Input [B, S, H] (Replicate)                         │
│       │                                              │
│       ▼                                              │
│  ┌─────────┬─────────┬─────────┐                     │
│  │ W_0     │ W_1     │ W_2     │  ← Weight 列切分    │
│  │ [H,H/3] │ [H,H/3] │ [H,H/3] │                     │
│  └────┬────┴────┬────┴────┬────┘                     │
│       ▼         ▼         ▼                          │
│  ┌─────────┬─────────┬─────────┐                     │
│  │ Y_0     │ Y_1     │ Y_2     │  ← Partial 输出     │
│  └─────────┴─────────┴─────────┘                     │
│       │ all-reduce                                   │
│       ▼                                              │
│  Output [B, S, H]                                    │
└──────────────────────────────────────────────────────┘
```

### 7.3 PyTorch TP vs Megatron TP 对比

| 特性 | PyTorch TP | Megatron TP |
|------|-----------|-------------|
| 抽象层 | DTensor + ParallelStyle | ColumnParallelLinear / RowParallelLinear |
| 通信 | 自动插入 all-reduce | 手动 f()/g() 函数（fused） |
| 设备支持 | DeviceMesh（设备无关） | 仅 GPU（CUDA） |
| 与 PP 集成 | PipelineStage | 原生 InterleavedSchedule |
| 激活重算 | 通过 DTensor 重计算 | 内置 recompute_granularity |

---

## 7. Pipeline Parallel

Pipeline Parallel（PP）将模型按层切分到多个设备，通过微批次（microbatch）流水线实现并行。PyTorch 2.x 推出了新的 `torch.distributed.pipelining` API。

### 8.1 PipelineStage

**概念原理**：`PipelineStage` 是 PP 的基本单元，封装一段连续的模型层。每个 stage 只持有部分层的参数，通过 `recv_forward` / `send_forward` 在 stage 间传递激活。

```python
# torch/distributed/pipelining/stage.py
class PipelineStage(Module):
    def __init__(self, submodule, stage_index, num_stages,
                 device, input_args=None, output_args=None, ...):
        self.model_partition = submodule
        self.stage_index = stage_index
        # ... 初始化通信配置

    def forward(self, args):
        # 接收上游输入 → 计算 → 发送到下游
        ...
```

### 8.2 调度策略

**概念原理**：PP 调度策略决定微批次的执行顺序。`Schedule1F1B` 是经典调度：先填充流水线（warmup），然后交替执行前向和反向（steady state），最后排空流水线（cooldown）。`ScheduleInterleaved1F1B` 在 interleaved PP（每个 stage 持有多个非连续块）上使用。

```python
# torch/distributed/pipelining/schedules.py:1020
class Schedule1F1B(_PipelineSchedule):
    # 1F1B 调度：warmup → steady state (1F1B 交替) → cooldown
    ...

# torch/distributed/pipelining/schedules.py:264
class _PipelineSchedule:
    def __init__(self, stages, n_microbatches, ...):
        self.stages = stages
        self.n_microbatches = n_microbatches
```

#### 1F1B 流水线时序图

```
时间 →
Stage 0: [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
Stage 1:      [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
Stage 2:           [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]
Stage 3:                [F0] [F1] [F2] [F3] [B0] [B1] [B2] [B3]

F = Forward, B = Reverse, 数字 = microbatch index
```

### 8.3 Microbatch 切分

```python
# 将输入 batch 切分为 microbatches
microbatches = split_tensor_into_1d_equal_chunks(input, chunks=n_microbatches)
```

---

## 8. DCP（Distributed Checkpoint）

`torch.distributed.checkpoint`（DCP）是 PyTorch 2.x 推出的新一代 checkpoint 系统，支持分布式环境下的高效保存/加载，是 FSDP2 推荐的 checkpoint 方案。

### 9.1 核心 API

**概念原理**：DCP 通过 `get_state_dict` / `set_state_dict` 操作嵌套的 state dict 结构，而非 FSDP1 的扁平化方式。`DefaultSavePlanner` 和 `DefaultLoadPlanner` 处理分片元数据，支持跨不同并行度的 checkpoint 转换。

```python
# torch/distributed/checkpoint/state_dict.py:1271
def get_state_dict(model, optimizer=None, *, submodules=None, ...):
    # 递归收集模型 + 优化器状态
    # 返回嵌套 dict: {"model": {...}, "optimizer": {...}}
    ...

# torch/distributed/checkpoint/state_dict.py:1481
def set_state_dict(model, optimizer=None, ...):
    # 从嵌套 dict 恢复模型 + 优化器状态
    ...
```

### 9.2 DCP 保存/加载流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DCP Checkpoint 流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Save:                                                              │
│  model + optimizer → get_state_dict() → nested dict                 │
│    → DefaultSavePlanner.plan() → 分片元数据                         │
│    → FileSystemWriter.write_data() → 分布式文件系统                  │
│                                                                     │
│  Load:                                                              │
│  FileSystemReader.read_data() → 分片数据                            │
│    → DefaultLoadPlanner.plan() → 映射到当前分片                      │
│    → set_state_dict() → model + optimizer                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.3 DCP vs 上层框架 Checkpoint 对比

| 特性 | PyTorch DCP | Megatron Checkpoint | DeepSpeed Checkpoint |
|------|------------|--------------------|--------------------|
| 状态结构 | 嵌套 dict | 扁平化（model/optimizer 分离） | 扁平化 |
| 分片感知 | 原生支持 | 需手动处理 | ZeRO 感知 |
| 跨并行度 | 支持（resharding） | 有限 | 有限 |
| 优化器状态 | 与模型统一 | 分离保存 | 分离保存 |

---

## 9. CUDA 基础设施

`torch.cuda` 提供 CUDA 运行时封装，包括 CUDAGraph（图捕获/回放）、CUDACachingAllocator（内存分配器）、Stream/Event（异步执行）等。

### 10.1 CUDAGraph

**概念原理**：CUDAGraph 将一系列 CUDA kernel 捕获为图（graph），回放时直接执行图而非逐个 kernel 启动，大幅降低 CPU 开销。适用于静态计算图（如固定 batch size 的训练循环）。

```python
# torch/cuda/graphs.py:390
def capture_begin(self, pool=None, mode=cudaStreamCaptureModeGlobal):
    # 开始捕获：cudaStreamBeginCapture(stream, mode)
    ...

# torch/cuda/graphs.py:447
def capture_end(self):
    # 结束捕获：cudaStreamEndCapture → cudaGraphInstantiate
    ...

# torch/cuda/graphs.py:485
def replay(self):
    # 回放：cudaGraphLaunch(graphExec, stream)
    ...

# torch/cuda/graphs.py:79
def is_current_stream_capturing():
    # 检查当前流是否正在捕获
    ...
```

#### CUDAGraph 捕获/回放流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CUDAGraph 生命周期                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. warmup: 空跑几次（让 CachingAllocator 稳定）                      │
│  2. capture_begin()  → cudaStreamBeginCapture                       │
│  3. 执行训练 kernel（被记录到图）                                     │
│  4. capture_end()    → cudaStreamEndCapture → cudaGraphInstantiate  │
│  5. replay()         → cudaGraphLaunch（低开销回放）                  │
│                                                                     │
│  注意：捕获期间禁止：                                                 │
│  - 动态 shape                                                        │
│  - CPU-GPU 同步                                                      │
│  - 内存分配/释放                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 CUDACachingAllocator

**概念原理**：`CUDACachingAllocator` 是 PyTorch 的 GPU 内存分配器，通过维护内存池避免频繁 `cudaMalloc` / `cudaFree`。内存按块（block）管理，支持分裂（split）和合并（coalesce）。

```python
# torch/cuda/memory.py:217
def empty_cache():
    # 释放所有缓存的未使用块
    # CUDACachingAllocator::emptyCache() → cudaFree
    ...

# torch/cuda/memory.py:232
def memory_stats(device=None):
    # 返回内存统计：allocated_bytes, active_bytes, reserved_bytes, ...
    ...
```

### 10.3 Stream 与 Event

```python
# torch/cuda/streams.py
class Stream:
    def __init__(self, device=None, priority=0, ...):
        # 创建 CUDA stream
        ...
    def wait_event(self, event): ...
    def wait_stream(self, stream): ...

class Event:
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        ...
    def record(self, stream=None): ...
    def wait(self, stream=None): ...
```

### 10.4 CUDA 基础设施 vs 上层框架对比

| 特性 | PyTorch CUDA | Megatron CUDA | DeepSpeed CUDA |
|------|-------------|--------------|----------------|
| 图捕获 | CUDAGraph | 不直接使用 | 不直接使用 |
| 内存分配 | CachingAllocator | 自定义 memory buffer | 自定义 allocator |
| 自定义 CUDA | 通过 extension | 大量 fused kernel | 大量 CUDA op |
| 与 NPU 兼容 | 不兼容（需 torch_npu） | 不兼容 | 不兼容 |

---

## 10. torch.compile（Inductor 后端）

`torch.compile` 是 PyTorch 2.x 的编译入口，通过捕获计算图并交给后端编译器（默认 Inductor）生成优化后的 kernel，实现性能提升。

### 11.1 编译流程

**概念原理**：`torch.compile` 首先通过 `TorchDynamo`（基于 Python bytecode 分析）捕获 FX 图，然后将 FX 图交给 `AOTAutograd` 分解为 forward/backward 图，最后 `Inductor` 将图编译为 Triton/C++ kernel。

```python
# torch/_inductor/compile_fx.py:857
def compile_fx_inner(model_, inputs_, ...):
    # 编译 FX 图的核心函数
    # 1. AOTAutograd 分解 fwd/bwd
    # 2. Inductor codegen
    ...

# torch/_inductor/compile_fx.py:937
def _compile_fx_inner(model_, inputs_, ...):
    # 实际编译逻辑
    ...
```

#### torch.compile 调用链

```
torch.compile(model)
  → Optimizer._compile / torch._dynamo 入口
    → TorchDynamo 捕获 Python bytecode → FX Graph
      → AOTAutograd 分解 forward/backward
        → Inductor 后端
          → lowerings → Triton kernel / C++ kernel
          → codegen → 生成优化代码
          → compile_fx_inner → 可执行函数
```

### 11.2 Inductor 关键概念

| 概念 | 作用 |
|------|------|
| `FxGraphCache` | FX 图缓存，避免重复编译 |
| `lowerings` | 将 ATen op 映射到 Inductor IR |
| `Triton kernel` | 自动生成 GPU kernel |
| `cpp kernel` | 自动生成 CPU kernel |
| `graph partitioner` | 将大图拆分为可编译子图 |

### 11.3 torch.compile vs 上层框架编译

| 特性 | torch.compile | Megatron | DeepSpeed |
|------|--------------|----------|-----------|
| 编译粒度 | 函数级（per-module） | 不编译 | 不编译 |
| 后端 | Inductor（Triton/C++） | 手动 fused kernel | 手动 CUDA op |
| 动态 shape | 支持（有开销） | 不适用 | 不适用 |
| 与 FSDP 集成 | 原生支持 | 不适用 | 不适用 |

---

## 11. multiprocessing（多进程启动）

`torch.multiprocessing` 封装 Python 标准库 `multiprocessing`，提供适合 PyTorch 训练的多进程启动方案。

### 12.1 spawn / start_processes

**概念原理**：`spawn` 是 PyTorch 推荐的启动方式——创建新进程并重新导入主模块，避免 fork 导致的 CUDA 上下文问题。`start_processes` 是更高级的封装，自动处理进程组和异常传播。

```python
# torch/multiprocessing/__init__.py
from torch.multiprocessing import spawn, start_processes

# spawn 用法
def worker(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    model = MyModel().cuda(rank)
    ...

spawn(worker, args=(world_size,), nprocs=world_size, join=True)
```

### 12.2 共享策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `file_system` | 基于文件系统共享 | 大 tensor |
| `file_descriptor` | 基于文件描述符共享 | 默认策略 |
| `shared_memory` | 基于 `/dev/shm` | 小 tensor |

### 12.3 torchrun 入口

```python
# torch/distributed/run.py
# torchrun 是 PyTorch 的分布式启动 CLI
# 自动设置 MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE
# 调用 start_processes 启动训练脚本
```

---

## 12. 跨模块调用链总图

下图展示一个完整训练步骤中各核心模块的调用关系：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          完整训练步骤调用链                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │ multiprocess│    │ distributed │    │   torch.nn  │    │  autograd   │       │
│  │   spawn     │───→│ init_proc   │───→│   Module    │───→│  backward   │       │
│  │             │    │ ess_group   │    │  .forward   │    │             │       │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘       │
│       │                  │                  │                    │              │
│       │                  │                  │                    │              │
│       ▼                  ▼                  ▼                    ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   torchrun  │    │   FSDP2     │    │   DTensor   │    │   optim     │       │
│  │             │    │ all_gather  │    │    TP/SP    │    │  .step()    │       │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘       │
│       │                  │                  │                    │              │
│       │                  │                  │                    │              │
│       ▼                  ▼                  ▼                    ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │   CUDA      │    │   DCP       │    │  Pipeline   │    │  compile    │       │
│  │  CUDAGraph  │    │ checkpoint  │    │   Stage     │    │  Inductor   │       │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. 关键配置参数表

### 14.1 FSDP2 配置参数

| 参数 | 默认值 | 含义 | 影响 |
|------|--------|------|------|
| `reshard_after_forward` | True | 前向结束后是否重新分片 | False 可减少通信但增加显存 |
| `mp_policy` | None | 混合精度策略 | 控制参数/梯度精度 |
| `mesh` | None | DeviceMesh | 定义并行拓扑 |
| `offload_policy` | None | CPU offload 策略 | 将参数卸载到 CPU |

### 14.2 DTensor / TP 配置参数

| 参数 | 默认值 | 含义 | 影响 |
|------|--------|------|------|
| `mesh_shape` | 必需 | 设备网格形状 | 决定并行度 |
| `Shard(dim)` | 必需 | 切分维度 | 决定数据分布方式 |
| `ColwiseParallel` | - | 列切分线性层 | 输出需 all-reduce |
| `RowwiseParallel` | - | 行切分线性层 | 输入需 all-reduce |

### 14.3 Pipeline Parallel 配置参数

| 参数 | 默认值 | 含义 | 影响 |
|------|--------|------|------|
| `n_microbatches` | 必需 | 微批次数量 | 影响流水线效率 |
| `split_points` | None | 模型切分点 | 决定 stage 边界 |
| `schedule` | 1F1B | 调度策略 | 影响显存和吞吐 |

### 14.4 torch.compile 配置参数

| 参数 | 默认值 | 含义 | 影响 |
|------|--------|------|------|
| `mode` | "default" | 编译模式 | reduce-overhead 启用 CUDAGraph |
| `fullgraph` | False | 是否要求完整图 | True 时禁止 graph break |
| `dynamic` | None | 动态 shape 模式 | True 支持动态 shape 但有开销 |
| `backend` | "inductor" | 编译后端 | 可选 cudagraphs、aot_eager 等 |

### 14.5 Optimizer 配置参数

| 参数 | 默认值 | 含义 | 影响 |
|------|--------|------|------|
| `lr` | 1e-3 | 学习率 | 收敛速度 |
| `weight_decay` | 1e-2 | 权重衰减 | 正则化强度 |
| `betas` | (0.9, 0.999) | Adam 动量参数 | 梯度平滑程度 |
| `foreach` | None | 批量实现 | 加速多参数更新 |
| `fused` | None | CUDA fused 实现 | 减少 kernel launch 开销 |

---

## 14. PyTorch 训练栈 vs 上层框架对比表

### 15.1 并行策略对比

| 并行维度 | PyTorch 原生 | Megatron-LM | DeepSpeed | torchtitan |
|---------|-------------|-------------|-----------|------------|
| 数据并行 | FSDP2 | DistributedOptimizer | ZeRO-1/2/3 | FSDP2 |
| 张量并行 | DTensor TP | Column/RowParallelLinear | 不支持 | DTensor TP |
| 流水线并行 | PipelineStage | InterleavedSchedule | PipeSchedule | PipelineStage |
| 专家并行 | 不支持 | 内置 | 不支持 | 不支持 |
| 上下文并行 | 不支持 | 内置 | 不支持 | 不支持 |

### 15.2 内存优化对比

| 技术 | PyTorch 原生 | Megatron-LM | DeepSpeed |
|------|-------------|-------------|-----------|
| 参数分片 | FSDP2 | DistributedOptimizer | ZeRO-2/3 |
| 梯度检查点 | `checkpoint` | `recompute_granularity` | `activation_checkpointing` |
| 优化器状态分片 | FSDP2 + offload | DistributedOptimizer | ZeRO-1/2/3 |
| CPU offload | FSDP offload_params | 不支持 | ZeRO-Offload |
| 激活重算 | `checkpoint` | selective recompute | uniform recompute |

### 15.3 编译与执行对比

| 特性 | PyTorch 原生 | Megatron-LM | DeepSpeed |
|------|-------------|-------------|-----------|
| 编译 | torch.compile (Inductor) | 手动 fused kernel | 手动 CUDA op |
| CUDA Graph | CUDAGraph / reduce-overhead | 不直接使用 | 不直接使用 |
| 混合精度 | AMP (autocast + GradScaler) | FP16/BF16 O1-O3 | AMP / ZeRO 内置 |
| 通信优化 | NCCL 内置 | 自定义 overlap | 自定义 overlap |

### 15.4 Checkpoint 对比

| 特性 | PyTorch 原生 | Megatron-LM | DeepSpeed |
|------|-------------|-------------|-----------|
| 保存格式 | DCP (nested dict) | 自定义（model/optimizer 分离） | 自定义 |
| 分片感知 | 原生 | 需手动 | ZeRO 感知 |
| 跨并行度 | 支持 resharding | 有限 | 有限 |
| 异步保存 | 支持 | 不支持 | 支持 |

---

## 附录：核心模块源码索引

| 模块 | 关键文件 | 核心类/函数 |
|------|---------|------------|
| `torch.nn` | `torch/nn/modules/module.py` | `Module.__call__` (:1919), `state_dict` (:2196) |
| `torch.nn` | `torch/nn/parameter.py` | `Parameter` (:30), `_ParameterMeta` (:19) |
| `torch.autograd` | `torch/autograd/__init__.py` | `backward` (:255), `_engine_run_backward` (:395) |
| `torch.autograd` | `torch/autograd/function.py` | `Function` (:364) |
| `torch.optim` | `torch/optim/optimizer.py` | `Optimizer` (:339), `step` (:1093) |
| `torch.optim` | `torch/optim/adamw.py` | `AdamW` (:20) |
| `torch.distributed` | `torch/distributed/distributed_c10d.py` | `init_process_group` (:2350), `all_reduce` (:3700) |
| `torch.distributed.fsdp` | `torch/distributed/fsdp/fully_sharded_data_parallel.py` | `FullyShardedDataParallel` (:397) |
| `torch.distributed.fsdp` | `torch/distributed/fsdp/_flat_param.py` | `FlatParamHandle` (:481) |
| `torch.distributed.tensor` | `torch/distributed/tensor/_api.py` | `DTensor` (:356), `distribute_tensor` (:978) |
| `torch.distributed.tensor` | `torch/distributed/tensor/placement_types.py` | `Shard`, `Replicate`, `Partial` |
| `torch.distributed.tensor` | `torch/distributed/tensor/parallel/api.py` | `parallelize_module` (:14) |
| `torch.distributed.pipelining` | `torch/distributed/pipelining/stage.py` | `PipelineStage` |
| `torch.distributed.pipelining` | `torch/distributed/pipelining/schedules.py` | `Schedule1F1B` (:1020) |
| `torch.distributed.checkpoint` | `torch/distributed/checkpoint/state_dict.py` | `get_state_dict` (:1271), `set_state_dict` (:1481) |
| `torch.cuda` | `torch/cuda/graphs.py` | `capture_begin` (:390), `replay` (:485) |
| `torch.cuda` | `torch/cuda/memory.py` | `empty_cache` (:217), `memory_stats` (:232) |
| `torch._inductor` | `torch/_inductor/compile_fx.py` | `compile_fx_inner` (:857), `_compile_fx_inner` (:937) |
| `torch.multiprocessing` | `torch/multiprocessing/__init__.py` | `spawn`, `start_processes` |
| `torch.distributed` | `torch/distributed/device_mesh.py` | `init_device_mesh` (:1544) |

---

## 附录 B：工作实战要点速查

| 场景 | 查哪里 | 关键代码 |
|------|--------|---------|
| 启用 FSDP2 | `fully_shard()` | `fully_sharded_data_parallel.py:397` |
| 配置 TP（Tensor Parallel） | `parallelize_module()` + `ColwiseParallel` | `tensor/parallel/api.py:14` |
| Pipeline Parallel | `PipelineStage` | `pipelining/stage.py` |
| DCP Checkpoint 保存 | `dcp.save()` / `dcp.load()` | `distributed/checkpoint/` |
| torch.compile 调试 | `compile_fx_inner()` | `_inductor/compile_fx.py:857` |
| CUDA Graph 捕获 | `CUDAGraph.capture_begin()` | `cuda/graphs.py:390` |
| 自定义 autograd Function | `Function.forward()` / `backward()` | `autograd/function.py:364` |
| DeviceMesh 创建 | `init_device_mesh()` | `distributed/device_mesh.py:1544` |
| 分布式初始化 | `init_process_group()` | `distributed_c10d.py:2350` |
| 混合精度训练 | `autocast` + `GradScaler` | `torch/amp/` |
| 多进程启动 | `mp.spawn()` | `multiprocessing/__init__.py` |
| 内存分析 | `memory_stats()` / `memory_snapshot()` | `cuda/memory.py:232` |

---

---

## 附录 C：常见坑与解决方案

| 问题现象 | 根因 | 解决方案 | 代码位置 |
|---------|------|---------|---------|
| FSDP2 resharding 慢 | 跨节点 AllGather 带宽不足 | 启用 `forward_prefetch=True` | `fully_sharded_data_parallel.py` |
| DTensor TP 报错 | mesh 维度与 TP 度不匹配 | 检查 `DeviceMesh` 形状 | `tensor/parallel/api.py` |
| torch.compile 图断裂 | 动态控制流 / 数据依赖 | 使用 `fullgraph=True` 或标记 `torch._dynamo.mark_static` | `_inductor/compile_fx.py` |
| CUDA Graph replay 错误 | 输入地址变化 | 固定输入 tensor 地址 | `cuda/graphs.py` |
| DCP 加载 shape 不匹配 | 并行度变化导致分片不同 | 使用 `reshard=True` | `distributed/checkpoint/` |
| autograd 内存泄漏 | 未释放中间 tensor | 使用 `torch.no_grad()` 或 `del` | `autograd/__init__.py` |

> **交叉引用**：上层框架如何使用 PyTorch 详见 `skill-knowledge/pretraining.md`（Megatron/torchtitan）、`skill-knowledge/deepspeed.md`（DeepSpeed）。

---

> **文档统计**：本文档覆盖 PyTorch 11 个核心模块，包含 60+ 处 file:line 源码引用、6 条完整调用链、4 幅 ASCII 架构图、5 张跨框架对比表、5 张配置参数表、3 个附录。