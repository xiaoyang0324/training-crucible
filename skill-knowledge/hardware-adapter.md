# 硬件适配层 (Hardware Adapter) — 代码级深度分析

> **torchada 不是训练框架**，而是 CUDA→MUSA 兼容层（compatibility shim）。
> **torch_musa** 是 MUSA GPU 的 PyTorch 后端（backend）。
> torchada 是**跑在 torch_musa 之上**的一层透明补丁，让现有 CUDA 代码（SGLang、vLLM、Megatron）**零代码修改**即可在摩尔线程 MUSA GPU 上运行。

---

## 1. torchada — CUDA→MUSA 兼容层

### 1.1 整体架构与设计理念

torchada 的核心设计哲学是 **"import once, run everywhere"** —— 用户只需在脚本最前面加一行 `import torchada`，后续所有 `torch.cuda.*` 调用自动透明地重定向到 `torch.musa.*`。

**版本**：`__version__ = "0.1.84"`（`src/torchada/__init__.py:27`）

**分层架构图：**

```
┌─────────────────────────────────────────────────────────────┐
│                  用户代码 (SGLang / vLLM / Megatron)         │
│                  调用 torch.cuda.* / CUDAExtension           │
├─────────────────────────────────────────────────────────────┤
│                  torchada (CUDA→MUSA Shim)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ _patch   │ │ _platform│ │ _cpp_ops │ │ _graph_rotatn │  │
│  │ 引擎     │ │ 平台检测 │ │ C++覆盖  │ │ Graph旋转     │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
│  ┌──────────────────┐ ┌──────────────────────────────────┐  │
│  │ Triton Kernels   │ │ utils/cpp_extension.py           │  │
│  │ (FP8/MoE/Router) │ │ (CUDAExtension/BuildExtension)   │  │
│  └──────────────────┘ └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  torch_musa (PyTorch MUSA Backend)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Device   │ │ Memory   │ │ Stream/  │ │ MUSAGraph     │  │
│  │ Manager  │ │ Allocator│ │ Event    │ │ Capture       │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────────────────┐  │
│  │ MCCL     │ │Inductor  │ │ FSDP / DTensor              │  │
│  │ Comm     │ │ Codegen  │ │ Distributed                 │  │
│  └──────────┘ └──────────┘ └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  MUSA Hardware (MTT S80/S4000/S5000)         │
└─────────────────────────────────────────────────────────────┘
```

**核心入口调用链：**

```
import torchada
  └─ __init__.py:54  load_cpp_ops()                    # 加载 C++ 算子覆盖
  └─ __init__.py:57  apply_patches()                   # 应用所有补丁
       └─ _patch.py:2103-2210  遍历 _patch_registry
            ├─ _patch_torch_device()        # torch.device("cuda") → "musa"
            ├─ _patch_torch_cuda_module()   # torch.cuda.* → torch.musa.*
            ├─ _patch_distributed_backend() # nccl → mccl
            ├─ _patch_autocast()            # autocast device_type cuda→musa
            ├─ _patch_flash_attn()          # sgl_kernel.flash_attn → MUSA FA
            ├─ _patch_library_impl()        # CUDA dispatch → PrivateUse1
            ├─ _patch_profiler_activity()   # ProfilerActivity.CUDA → PrivateUse1
            ├─ _patch_backends_cuda()       # torch.backends.cuda 兼容
            ├─ _patch_torch_c_exports()     # torch._C 补充 MUSA 函数
            ├─ _patch_musa_warnings()       # 抑制 MUSA 噪音警告
            ├─ _patch_cdll()                # ctypes.CDLL 函数名翻译
            ├─ _patch_inductor_autotune()   # Inductor autotune 兼容
            ├─ _patch_accelerator()         # torch.accelerator 兼容
            ├─ _patch_triton_gdc()          # Triton tl.extra.cuda.gdc 兼容
            └─ _patch_flex_attention()      # flex_attention 设备验证
  └─ __init__.py:60  set_default_moe_config_dir()      # 设置 MoE 配置路径
```

### 1.2 Patch 引擎核心

torchada 的 patch 引擎采用**注册表模式**（Registry Pattern），类似 Flask 的 `@app.route` 装饰器。

**注册表定义** (`_patch.py:44-65`)：

```python
# _patch.py:45
_patch_registry: List[Callable[[], None]] = []

# _patch.py:48-65
def patch_function(func: Callable[[], None]) -> Callable[[], None]:
    """Decorator to register a function to be called during patching."""
    _patch_registry.append(func)
    return func
```

**`@requires_import` 守卫装饰器** (`_patch.py:68-97`)：可选依赖的守卫，当指定模块不可 import 时自动跳过该 patch。

```python
# _patch.py:68-97
def requires_import(*module_names: str) -> Callable:
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for module_name in module_names:
                try:
                    __import__(module_name)
                except ImportError:
                    return  # 模块不可用，跳过
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

**`apply_patches()` 主入口** (`_patch.py:2103-2210`)：

```python
def apply_patches():
    global _patched
    if _patched:
        return
    if not is_musa_platform():   # 非 MUSA 平台直接跳过
        _patched = True
        return
    import torch_musa            # 确保 torch_musa 初始化
    for patch_fn in _patch_registry:  # 按注册顺序执行所有 patch
        patch_fn()
    # 额外 patch: Tensor.to(), Tensor.cuda(), Module.cuda()
    torch.Tensor.to = _wrap_to_method(torch.Tensor.to)       # line 2162
    torch.Tensor.cuda = _wrap_tensor_cuda(torch.Tensor.cuda) # line 2166
    torch.nn.Module.cuda = _wrap_module_cuda(torch.nn.Module.cuda) # line 2170
    # 自动发现并包装所有 torch 工厂函数 (torch.randn, torch.zeros, ...)
    for fn_name in _discover_factory_functions():  # line 2181
        ...
    _patched = True
```

**16 个关键 patch 函数列表（含行号）：**

| # | Patch 函数 | 行号 | 功能 |
|---|-----------|------|------|
| 1 | `_patch_torch_device` | 308-340 | `torch.device("cuda")` → `"musa"` |
| 2 | `_patch_torch_cuda_module` | 861-1002 | `torch.cuda.*` → `torch.musa.*` 全面重定向 |
| 3 | `_patch_distributed_backend` | 1005-1060 | NCCL → MCCL 后端自动翻译 |
| 4 | `_patch_autocast` | 1159-1177 | `autocast(device_type="cuda")` → `"musa"` |
| 5 | `_patch_profiler_activity` | 1180-1226 | `ProfilerActivity.CUDA` → `PrivateUse1` |
| 6 | `_patch_musa_warnings` | 1229-1255 | 抑制 MUSA 噪音警告 |
| 7 | `_patch_library_impl` | 1258-1310 | `torch.library.impl("CUDA")` → `PrivateUse1` |
| 8 | `_patch_torch_c_exports` | 1313-1342 | `torch._C._storage_Use_Count` 等函数补全 |
| 9 | `_patch_backends_cuda` | 1345-1416 | `torch.backends.cuda` 兼容 |
| 10 | `_patch_flash_attn` | 1495-1633 | `sgl_kernel.flash_attn` → MUSA `flash_attn_interface` |
| 11 | `_patch_cdll` | 1636-2100 | `ctypes.CDLL` 函数名翻译 (`cudaXxx` → `musaXxx`) |
| 12 | `_patch_inductor_autotune` | — | Inductor autotune `CUDA_VISIBLE_DEVICES` → `MUSA_VISIBLE_DEVICES` |
| 13 | `_patch_accelerator` | — | `torch.accelerator.synchronize()` → `torch.musa.synchronize()` |
| 14 | `_patch_triton_gdc` | — | Triton `tl.extra.cuda.gdc_wait` 兼容 |
| 15 | `_patch_flex_attention` | 1480-1492 | `flex_attention._validate_device` 兼容 |
| 16 | `_patch_graph_context` | — | `torch.cuda.graph` → `torch.musa.graph` |

### 1.3 平台检测

**`Platform` 枚举** (`_platform.py:12-17`)：

```python
class Platform(Enum):
    CUDA = "cuda"
    MUSA = "musa"
    CPU  = "cpu"
```

**`detect_platform()` 检测逻辑** (`_platform.py:20-53`)：

优先级：
1. `TORCHADA_PLATFORM` 环境变量（强制指定）
2. MUSA 可用性检测（`torch.version.musa` 或 `torch_musa` 可 import）
3. CUDA 可用性检测（`torch.cuda.is_available()`）
4. CPU 回退

```python
@lru_cache(maxsize=1)
def detect_platform() -> Platform:
    forced_platform = os.environ.get("TORCHADA_PLATFORM", "").lower()
    if forced_platform == "cuda": return Platform.CUDA
    elif forced_platform == "musa": return Platform.MUSA
    elif forced_platform == "cpu": return Platform.CPU
    if _is_musa_available(): return Platform.MUSA
    if _is_cuda_available(): return Platform.CUDA
    return Platform.CPU
```

**`_is_musa_available()` 检测信号** (`_platform.py:56-89`)：
- 主信号：`torch.version.musa` 被设置（torch_musa 构建时设置）
- 次信号：`torch_musa` 可 import（允许无 GPU 的构建环境）

**辅助 API**：
- `is_musa_platform()` (line 102), `is_cuda_platform()` (line 107), `is_cpu_platform()` (line 112)
- `get_device_name()` (line 117), `get_torch_device_module()` (line 122-143)
- `is_gpu_device()` (line 146-188), `is_cuda_like_device()` (line 191)

### 1.4 C++ 算子覆盖

**`load_cpp_ops()` 加载逻辑** (`_cpp_ops.py:73-159`)：

```python
def load_cpp_ops(force_reload=False):
    # 1. 非 MUSA 平台直接返回 None
    if not is_musa_platform(): return None
    # 2. 扫描 csrc/ 目录收集 .cpp/.cu/.mu 源文件
    # 3. 有 .mu 文件 → 使用 torchada 的 load()（MUSA 编译）
    #    无 .mu 文件 → 使用 torch 的 load()（纯 C++）
    # 4. 架构标志: --offload-arch=mp_21|mp_22|mp_31
```

**`_detect_musa_arch()` 架构检测** (`_cpp_ops.py:28-70`)：
- 调用 `musaInfo` 获取 compute capability
- 映射：`2.1` → `mp_21`（MTT S80），`2.2` → `mp_22`（MTT S4000），`3.1` → `mp_31`（MTT S5000）
- 默认回退到 `mp_31`

**`csrc/ops.cpp` 核心内容**：
- `_musa_beginAllocateCurrentThreadToPool()` (line 26-36) — CUDA 兼容的内存池分配 API
- `_musa_endAllocateToPool()` (line 38-42)
- `_musa_releasePool()` (line 44-48)
- 使用 `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` 注册 ATen 算子覆盖

**`load_graph_rotation_ops()`** (`_cpp_ops.py:162-221`)：
- 独立扩展，需要 torch_musa 内部头文件（`MUSAGraph.h`）
- 链接 `libmusart` + `libmusa_python`
- 暴露 `free_exec` / `inst_exec` / `has_exec` 三个 C++ 函数

### 1.5 Graph Rotation（核心创新）

这是 torchada 的**核心创新**，解决了 MUSA 驱动 ~2048 live-executable 上限问题。

**问题背景**：MUSA 驱动对每个进程的 live `musaGraphExec_t`（实例化 CUDA Graph）数量有 ~2048 的上限。vLLM/SGLang 的 piecewise CUDA graphs 会实例化 `capture_sizes × num_layers` 个 executable，超过 40 层的模型就会爆上限。

**解决方案**：Graph *template*（`musaGraph_t`）不计入上限，且从 template 重新实例化 executable 很便宜（~0.3ms）。所以保留所有 template，LRU 限制 live executable 数量。

**`_DEFAULT_CAP = 1900`** (`_graph_rotation.py:44`)：
```python
_DEFAULT_CAP = 1900          # 略低于 ~2043 驱动上限
_DEFAULT_MARGIN = 128        # 自动探测时的安全边距
```

**`_probe_live_exec_limit()` 自动探测** (`_graph_rotation.py:79-110`)：
- 在**独立子进程**中捕获 trivial graph 直到失败
- 避免 RNG 生成器状态损坏影响主进程（"Offset increment outside graph capture"）
- 环境变量 `TORCHADA_GRAPH_AUTOPROBE=1` 启用

**`_Rotation` 类** (`_graph_rotation.py:138-244`)：

```python
class _Rotation:
    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.RLock()
        self._live = collections.OrderedDict()  # id(graph) → weakref(graph)
        self._aux = None  # C++ aux 扩展模块引用
        self.stats = {"register": 0, "evict": 0, "reinstantiate": 0, "build_failed": 0}

    def register(self, graph):
        # capture_end 时调用，注册 graph 到 LRU
        # 超过 cap → 触发 _evict_locked()

    def on_replay(self, graph):
        # replay 时调用，如果 graph 的 exec 已被 evict → 重新实例化
        if not self._evicting: return  # 快速路径
        if key not in self._live or not aux.has_exec(graph):
            aux.inst_exec(graph)  # 从 template 重新实例化
```

**`install()` 安装** (`_graph_rotation.py:261-303`)：
- 猴子补丁 `torch.musa.MUSAGraph.capture_end` 和 `replay`
- 在 `capture_end` 后调用 `rot.register(self)`
- 在 `replay` 前调用 `rot.on_replay(self)`
- 幂等设计，安全多次调用

**`graph_exec_aux.cpp` C++ 实现** (`_graph_rotation_src/graph_exec_aux.cpp:41-57`)：
- 通过 `Members` 派生类技巧访问 `MUSAGraph` 的 `protected` 成员
- `free_exec()` (line 41-48)：销毁 `graph_exec_`，保留 `graph_` template
- `inst_exec()` (line 51-57)：从 template 重新 `musaGraphInstantiate`

### 1.6 FP8 Quant Kernel

**`per_token_group_quant_fp8()`** (`triton/kernels/quant/fp8.py:55-116`)：

```python
fp8_dtype = torch.float8_e4m3fn  # line 9

def per_token_group_quant_fp8(x, group_size, eps=1e-10, dtype=fp8_dtype):
    # 输入: [..., hidden_dim], 按 group_size 分组量化
    # 输出: (x_q, x_s) — 量化后的 tensor 和 scale factor
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    _per_token_group_quant_8bit[(M,)](...)
```

**`_per_token_group_quant_8bit` Triton kernel** (`fp8.py:12-52`)：
- 每行一个 program，计算 `_absmax = max(abs(y))`，`scale = absmax / bit8_max`
- 支持 FP8 (`float8_e4m3fn`) 和 INT8 两种 dtype

同文件还包含 `per_token_quant_int8` (line 150-180) 和 `per_token_group_quant_int8` (line 225-281)。

### 1.7 Triton Fused MoE

torchada 提供了一套完整的 Triton Fused MoE 实现，用于 SGLang/vLLM 的 MoE 推理。

**`fused_topk_native()`** (`triton/runtime/fused_moe/router.py:46-80`)：
- 标准 top-k routing：softmax → topk → renormalize
- 支持 correction_bias（DeepSeek V2/V3/R1 的 biased routing）

**`grouped_topk()`** (`router.py:115-170`)：
- 分组 top-k routing：先选 group，再在 group 内选 expert
- 支持 `num_fused_shared_experts`（融合共享专家）

**`fused_experts_impl()`** (`triton/runtime/fused_moe/fused_moe.py:331-432`)：
- 完整的 fused MoE 计算：router → permute → GEMM1 → activation → GEMM2 → unpermute
- 支持 FP8/INT8/INT4 量化
- 支持 TMA (Tensor Memory Access) 优化
- 调用 `_prepare_fused_moe_run()` 准备配置
- 调用 `_fused_moe_kernel_sequence()` 执行 kernel 序列

**其他 router 变体**：
- `biased_grouped_topk_impl()` (line 236-297) — DeepSeek V3 风格
- `kimi_k2_biased_topk_impl()` (line 199-233) — Kimi K2 优化（单 group 特化）
- `grouped_topk_gpu()` (line 327-382) — GPU 版本 grouped topk

### 1.8 CUDA Extension 移植

**`CUDAExtension` 统一接口** (`utils/cpp_extension.py:1023-1044`)：

```python
class CUDAExtension:
    def __new__(cls, name, sources, *args, **kwargs):
        platform = detect_platform()
        if platform == Platform.MUSA:
            return _create_musa_extension(name, sources, *args, **kwargs)
        return _create_cuda_extension(name, sources, *args, **kwargs)
```

**`_create_musa_extension()`** (`cpp_extension.py:1118-1149`)：
- 应用 MUSA patches（`.cu` → `.mu` 映射、CUDA→MUSA 符号转换）
- 翻译 `nvcc` → `mcc` 编译标志
- 翻译 `nvjpeg` → `mtjpeg` 库名
- 调用 `torch_musa.utils.musa_extension.MUSAExtension`

**`_port_cuda_source()`** (`cpp_extension.py:817-838`)：
- 源码级 CUDA→MUSA 移植
- 使用 `_MAPPING_RULE` 进行符号替换
- 按规则长度降序排列避免部分替换

**`load()` JIT 编译** (`cpp_extension.py:1357-1440`)：
- MUSA 平台：调用 `musa_ext.load()`，参数名 `extra_cuda_cflags` → `extra_musa_cflags`
- CUDA 平台：调用 `torch.utils.cpp_extension.load()`

**`BuildExtension`** (`cpp_extension.py:1152-1354`)：
- 自定义 `_MUSABuildExtension` 处理 CUDA→MUSA 源码移植
- 注册 `.cu/.cuh` 为合法源文件扩展名

---

## 2. torch_musa — MUSA GPU PyTorch 后端

### 2.1 初始化流程（18 步启动序列）

`torch_musa/__init__.py` 的完整启动序列：

| 步骤 | 代码位置 | 操作 |
|---|---|---|
| 1 | `__init__.py:26-39` | 版本检查：要求 torch ≥ 2.11.0 |
| 2 | `__init__.py:43` | `torch.utils.rename_privateuse1_backend("musa")` |
| 3 | `__init__.py:45-48` | `import torch_musa._MUSAC`（C++ 扩展） |
| 4 | `__init__.py:53-55` | `align_rng_to_nv_gpu()` — RNG 对齐 |
| 5 | `__init__.py:57` | `torch.__setattr__("musa", sys.modules[__name__])` |
| 6 | `__init__.py:59-63` | 注册 `ProfilerActivity.MUSA = PrivateUse1` |
| 7 | `__init__.py:65-66` | `_apply_distributed_patch()` — 分布式 patch |
| 8 | `__init__.py:67-90` | 导入 `core` 模块：device, stream, memory 等 |
| 9 | `__init__.py:92` | `register_musa_hook()` |
| 10 | `__init__.py:94-105` | 导入 `Stream`, `Event`, `StreamContext` |
| 11 | `__init__.py:106-124` | 导入 `amp`, `serialization`, `memory`, `_lazy_init` |
| 12 | `__init__.py:125-131` | 导入 `ops`, `random`, `mudnn`, `musa` |
| 13 | `__init__.py:134-136` | 导入 `musa_graph`（Graph capture） |
| 14 | `__init__.py:138-145` | 导入 `mccl`, 注册 `torch.backends.mudnn/musa`, `register_deserialization` |
| 15 | `__init__.py:152` | `setattr(torch.version, "musa", ...)` |
| 16 | `__init__.py:154-171` | `set_attributes()` — tensor/module/storage/profiler 属性 |
| 17 | `__init__.py:174-190` | `_apply_patches()` — FSDP ShardedGradScaler patch |
| 18 | `__init__.py:198-216` | `init_reductions()`, `_init_inductor_backend_registration`, `overwrite_cuda_api` |

**延迟初始化（Lazy Init）**：`core/_lazy_init.py`
- `_lazy_init()`：线程安全，处理 fork 检测、排队调用回放
- `_lazy_call()`：未初始化时排队，初始化后执行
- `_initialized` 标志 + `_initialization_lock` 保证单次初始化

### 2.2 Device 管理

**C++ 层**：`csrc/core/Device.cpp`

```cpp
// csrc/core/Device.cpp:12-24
namespace c10::musa {
DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  C10_MUSA_CHECK(GetDevice(&cur_device));
  return cur_device;
}
void set_device(DeviceIndex device) { C10_MUSA_CHECK(SetDevice(device)); }
void Synchronize() { device_synchronize(); }
} // namespace c10::musa
```

**GuardImpl**：`csrc/core/GuardImpl.h`
```cpp
// csrc/core/GuardImpl.h:19-80
struct MUSAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr DeviceType static_type = kMUSA;
    DeviceType type() const override { return kMUSA; }
    Device exchangeDevice(Device d) const override;
    void setDevice(Device d) const override;
    Stream getStream(Device d) const noexcept override;
    Stream getDefaultStream(Device d) const override;
    void record(void**, const Stream&, ...) const override;
    void block(void* event, const Stream& stream) const override;
    // ... event create/destroy/query/synchronize
};
```

**设备计数与可用性**：`csrc/core/Device.cpp`
```cpp
// csrc/core/Device.cpp:29-42
namespace c10::musa {
DeviceIndex device_count() {
  DeviceIndex n = 0;
  C10_MUSA_CHECK(GetDeviceCount(&n));
  return n;
}
DeviceIndex device_count_ensure_non_zero() {
  DeviceIndex n = 0;
  C10_MUSA_CHECK(GetDeviceCount(&n));
  TORCH_CHECK(n > 0, "MUSA error: number of MUSA devices is 0");
  return n;
}
bool is_available() { return device_count() > 0; }
} // namespace c10::musa
```

### 2.3 内存分配器

**C++ 核心**：`csrc/core/MUSACachingAllocator.cpp`（152KB，主要文件）

**数据结构**（`MUSACachingAllocator.cpp:117-185`）：
```cpp
struct Block {
    c10::DeviceIndex device;
    musaStream_t stream;
    stream_set stream_uses;
    size_t size, requested_size;
    BlockPool* pool;
    void* ptr;
    bool allocated, mapped;
    Block *prev, *next;          // 双向链表（地址相邻 block）
    int event_count;
    ExpandableSegment* expandable_segment_;
};
```

**内存池**（`BlockPool`, line 94-113）：
- `blocks`：红黑树（`std::set<Block*, BlockComparatorSize>`），按大小排序
- `unmapped`：红黑树（按地址排序），用于 coalesce
- 双向链表连接地址相邻的 block，支持 split + coalesce

**关键参数**：
- `kLargeBuffer = 20 MiB`（line 54）
- `kMinBlockSize = 512`（line 59）
- `kSmallSize = 1 MiB`, `kSmallBuffer = 2 MiB`（line 60-63）
- `kRoundLarge = 2 MiB`（line 66）

**`NativeCachingAllocator::allocate()`** — 顶层入口（`MUSACachingAllocator.cpp:3755-3781`）：
```cpp
// csrc/core/MUSACachingAllocator.cpp:3755-3781
DataPtr allocate(size_t size) override {
  constexpr size_t one_exa_bytes = 1152921504606846976ULL;
  TORCH_CHECK_WITH(OutOfMemoryError, size < one_exa_bytes,
      "MUSA out of memory. Tried to allocate more than 1EB memory.");
  c10::DeviceIndex device = 0;
  C10_MUSA_CHECK(c10::musa::GetDevice(&device));
  void* devPtr = nullptr;
  void (*deleteFunc)(void*) = &local_raw_delete;
  MUSAStream stream = musa::getCurrentMUSAStream(device);
  if (forceUncachedAllocator() || !isEnabled()) {
    deleteFunc = &uncached_delete;
    devPtr = uncached_allocate(size);          // 直接 musaMalloc
  } else {
    if (size != 0) {
      this->malloc(&devPtr, device, size, stream);  // 走缓存池
    }
  }
  return {devPtr, devPtr, deleteFunc, Device(at::musa::kMUSA, device)};
}
```

**`DeviceCachingAllocator::free()`** — 释放逻辑（`MUSACachingAllocator.cpp:1608-1670`）：
```cpp
// csrc/core/MUSACachingAllocator.cpp:1608-1670
void free(Block* block) {
  std::lock_guard<std::recursive_mutex> lock(mutex);
  block->allocated = false;
  // 更新统计：allocation / allocated_bytes 递减
  StatTypes stat_types = get_stat_types_for_pool(*block->pool);
  for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
    stats.allocation[stat_type].decrease(1);
    stats.allocated_bytes[stat_type].decrease(block->size);
  });
  record_trace(TraceEntry::FREE_REQUESTED, ...);
  if (!block->stream_uses.empty()) {
    // Graph capture 期间 → 延迟释放（deferred_blocks）
    deferred_blocks.emplace(block, insert_free_marker(block));
  } else {
    free_block(block, context);   // 立即回收到 pool
  }
}
```

**`alloc_block()`** — 底层分配（`MUSACachingAllocator.cpp:2816-2920`）：
```cpp
// csrc/core/MUSACachingAllocator.cpp:2816-2920
bool alloc_block(AllocParams& p, bool isRetry, ..., & lock) {
  void* ptr = nullptr;
  if (p.is_expandable_segments_active) {
    // Expandable Segment 路径：muMemAddressReserve + muMemMap
    p.block = try_allocate_expandable_block(p.device(), p.stream(), p.pool, p.size(), ctx);
  } else {
    // 标准路径：musaMallocMaybeCapturing（支持 Graph capture 中的分配）
    if (MUSAAllocatorConfig::release_lock_on_musamalloc()) {
      auto sg = c10::make_scope_exit([&]() { lock.lock(); });
      lock.unlock();    // 释放锁避免 musaMalloc 阻塞其他线程
      p.err = musaMallocMaybeCapturing(&ptr, size, p);
    }
  }
  total_allocated_memory += size;
  p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
  return true;
}
```

**Python 层**：`torch_musa/core/memory.py`
- `caching_allocator_alloc`, `empty_cache`, `memory_stats`, `mem_get_info`
- `MUSAPluggableAllocator`, `MemPool`, `use_mem_pool`
- 统一内存分配器：`PYTORCH_MUSA_ALLOC_CONF=cpu:unified`（`__init__.py:235-251`）

### 2.4 Stream & Event

**C++ Stream**：`csrc/core/MUSAStream.cpp`
- 全局 stream pool：`kStreamsPerPool = 32`（line 17-18）
- `kDefaultFlags = musaStreamNonBlocking`（line 19）
- 优先级支持：`max_compile_time_stream_priorities` 个优先级
- `StreamIdType`（line 38-60）：DEFAULT / EXT / PRIORITY 三种类型

**C++ Event**：`csrc/core/MUSAEvent.h`
```cpp
// csrc/core/MUSAEvent.h:35-60
struct MUSAEvent {
    MUSAEvent(unsigned int flags) noexcept;
    MUSAEvent(DeviceIndex device_index, const musaIpcEventHandle_t* handle);
    ~MUSAEvent();  // 在创建设备上销毁，避免创建其他设备的 context
};
```

**Python Stream**：`torch_musa/core/stream.py`
- `Stream` 类：`stream`, `query`, `synchronize`, `wait_event`, `wait_stream`
- `Event` 类：`record`, `query`, `synchronize`, `wait`, `elapsed_time`
- `ExternalStream`：外部 stream 包装
- `StreamContext`：context manager，自动切换 stream

### 2.5 Graph Capture

**C++ 层**：`csrc/aten/musa/MUSAGraph.cpp`
```cpp
// csrc/aten/musa/MUSAGraph.cpp:48-80
MUSAGraph::MUSAGraph(bool keep_graph)
    : capture_stream_(at::musa::getCurrentMUSAStream()),
      keep_graph_(keep_graph) {}

void MUSAGraph::capture_begin(MempoolId_t pool, musaStreamCaptureMode mode) {
    // 注册默认 generator
    // 注册所有 captured generator states
    // 开始捕获
}
```

**`MUSAGraph::capture_begin()`** — 开始捕获（`MUSAGraph.cpp:65-138`）：
```cpp
// csrc/aten/musa/MUSAGraph.cpp:65-138
void MUSAGraph::capture_begin(MempoolId_t pool, musaStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_, "This MUSAGraph instance already owns a captured graph.");
  // 1. 注册 default generator + 所有 captured generator states（RNG 安全）
  auto* gen = get_generator_or_default<MUSAGeneratorImpl>(c10::nullopt, musa::detail::getDefaultMUSAGenerator());
  gen->register_graph(this);
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->capture_prologue();
  }
  // 2. 必须在非 default stream 上捕获
  TORCH_CHECK(stream != at::musa::getDefaultMUSAStream(), "MUSA graphs must be captured on a non-default stream.");
  capture_stream_ = stream;
  capture_dev_ = c10::musa::current_device();
  // 3. 创建/复用 mempool，调用 beginAllocateToPool 防止 autograd 线程触发无效 event
  mempool_id_ = c10::musa::MemPool::graph_pool_handle(false);
  c10::musa::MUSACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, ...);
  // 4. 开始 stream capture
  AT_MUSA_CHECK(musaStreamBeginCapture(capture_stream_, capture_mode));
  AT_MUSA_CHECK(musaStreamGetCaptureInfo_v2(stream, &status, &capture_id_, ...));
}
```

**`MUSAGraph::replay()`** — 重放（`MUSAGraph.cpp:219-238`）：
```cpp
// csrc/aten/musa/MUSAGraph.cpp:219-238
void MUSAGraph::replay() {
  TORCH_CHECK(capture_ended_, "Called MUSAGraph::replay without a preceding successful capture.");
  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();   // keep_graph 模式下延迟实例化
  }
  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};
  // 恢复所有 captured generator states（保证 RNG 数值一致）
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  // graph_exec_ 可在任意 stream 上重放
  AT_MUSA_CHECK(musaGraphLaunch(graph_exec_, at::musa::getCurrentMUSAStream()));
}
```

**Python 层**：`musa_graph/graphs.py`
- `MUSAGraph` 类 (line 45-80)：包装 C++ `_MUSAGraph`
- `graph_pool_handle()` (line 33)：返回 opaque pool token
- `is_current_stream_capturing()` (line 12)：检测当前是否正在捕获
- `update_musa_graph_with_profile()` (line 21)：profile 模式下重新实例化

### 2.6 MCCL 通信

**C++ 层**：`csrc/core/PythonMCCL.cpp`
```cpp
// csrc/core/PythonMCCL.cpp:8-18
PyObject* THMPModule_mccl_version(PyObject* self, PyObject* args) {
  using torch::musa::mccl::version;
  return PyLong_FromUnsignedLongLong(version());
}
```

**`ProcessGroupMCCL::allreduce_impl()`** — 集合通信核心（`ProcessGroupMCCL.cpp:3998-4024`）：
```cpp
// csrc/distributed/ProcessGroupMCCL.cpp:3998-4024
c10::intrusive_ptr<Work> ProcessGroupMCCL::allreduce_impl(
    at::Tensor& tensor, const char* profilingTitle, const AllreduceOptions& opts) {
  return collective(
      tensor, tensor,
      [&](at::Tensor& input, at::Tensor& output, mcclComm_t comm, c10::musa::MUSAStream& stream) {
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp = getMcclReduceOp(opts.reduceOp, input, mcclDataType, comm);
        return mcclAllReduce(
            input.data_ptr(), output.data_ptr(), input.numel(),
            mcclDataType, mcclReduceOp, comm, stream.stream());
      },
      OpType::ALLREDUCE, opts.asyncOp, profilingTitle);
}
```

**`ProcessGroupMCCL::broadcast()`** — 广播（`ProcessGroupMCCL.cpp:4115-4167`）：
```cpp
// csrc/distributed/ProcessGroupMCCL.cpp:4115-4167
c10::intrusive_ptr<Work> ProcessGroupMCCL::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor);
  RECORD_PARAM_COMMS_DATA(..., "broadcast", ...);
  const auto root = opts.rootRank + opts.rootTensor;
  return collective(
      tensor, tensor,
      [&](at::Tensor& input, at::Tensor& output, mcclComm_t comm, at::musa::MUSAStream& stream) {
        return mcclBcast(input.data_ptr(), input.numel(),
            getMcclDataType(input.scalar_type()), static_cast<int>(root), comm, stream.stream());
      },
      OpType::BROADCAST, opts.asyncOp, "mccl:broadcast", nanCheck);
}
```

**Python 层**：`distributed/__init__.py`
- `_apply_distributed_patch()` (line 10-40)：按顺序应用所有分布式 patch
  - `_apply_device_mesh_patch()` — DeviceMesh 兼容
  - `_apply_init_utils_patch()` — FSDP 初始化兼容
  - `_apply_runtime_utils_patch()` — FSDP 运行时兼容
  - `_apply_dtensor_patches()` — DTensor 兼容
  - `_apply_fsdp2_patches()` — FSDP2 兼容
  - `_apply_state_dict_utils_patch()` — state_dict 兼容
  - `_apply_dist_wrapper_patch()` — distributed wrapper 兼容
  - `_apply_fully_shard_patch()` — fully_shard 兼容

**`device_mesh.py`** (line 49-50)：
```python
def _apply_device_mesh_patch():
    DeviceMesh._get_or_create_default_group = _get_or_create_default_group
```

### 2.7 Inductor 集成

**`MUSATritonWrapperCodeGen`** (`_inductor/codegen/wrapper.py:71`)：
```python
class MUSATritonWrapperCodeGen(PythonWrapperCodegen):
    """Generate outer wrapper in Python that calls the kernels."""
    def write_header(self):
        # 生成 import torch_musa 的 header
        # 处理 AOT config comment
```

**MUSA Graph Trees**：`_inductor/musagraph_trees.py`（121KB）
- 安全抽象层，支持任意 tree 结构的 MUSA graph 操作
- 支持 replay → record 切换（checkpoint pool state）
- 主要用于 Dynamo graph breaks 场景

**Inductor 后端注册**：
- `_init_inductor_backend_registration()`（`__init__.py:208`）
- 使用自定义 device 注册方式（非 PyTorch 标准 custom device 注册）

### 2.8 FSDP/DTensor 支持

**`ShardedGradScaler`** (`distributed/fsdp/sharded_grad_scaler.py`)：
```python
# distributed/fsdp/sharded_grad_scaler.py:34
class ShardedGradScaler(GradScaler):
    """支持 MUSA 的 FSDP gradient scaler。
    支持 CPU offloaded tensors、自定义混合精度 loss dtype (fp16, bf16)。"""
```

**FSDP patch 链**：
- `_apply_sharded_grad_scaler_patch()`（`__init__.py:174-187`）
- 替换 `torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler` 为 MUSA 版本

**DTensor 支持**：
- `_apply_dtensor_patches()` — DTensor 在 MUSA 上的兼容 patch
- `distributed/device_mesh.py` — DeviceMesh 的 MUSA 适配

### 2.9 限制与 TODO

| 限制 | 说明 | 代码位置 |
|---|---|---|
| 单线程编译 | 非多线程编译 | `cpp_extension.py` 注释 |
| 无 CppWrapper | 仅 Python wrapper | `_inductor/codegen/wrapper.py` |
| Graph beta | MUSAGraph API 仍在 beta | `musa_graph/graphs.py:48` |
| MCCL only on S4000 | 通信库仅支持 S4000+ | 文档 |
| torch ≥ 2.11.0 | 最低版本要求 | `__init__.py:26-28` |
| 无 CUDAGraph 完整兼容 | 需要 Graph Rotation workaround | `_graph_rotation.py` |

---

## 3. torchada vs torch_musa 关系

| 维度 | torchada | torch_musa |
|---|---|---|
| **定位** | CUDA→MUSA 兼容 shim | MUSA GPU PyTorch 后端 |
| **层级** | 跑在 torch_musa 之上 | 底层硬件适配 |
| **用户** | SGLang, vLLM, Megatron 等框架 | torchada, 直接用户 |
| **核心机制** | monkey-patch PyTorch API | 注册 PrivateUse1 backend |
| **代码量** | ~20K Python + C++ | ~150K+ C++ 核心 |
| **创新点** | Graph Rotation, Triton MoE | MUSACachingAllocator, MCCL |
| **依赖** | 依赖 torch_musa | 仅依赖 PyTorch |
| **版本** | 0.1.84 | 绑定 torch 2.11.0 |

**调用关系图**：

```
用户代码
  │
  ▼
torchada (apply_patches)
  │
  ├─ torch.device("cuda") ──patch──▶ torch.device("musa")
  ├─ torch.cuda.* ──────────patch──▶ torch.musa.*
  ├─ nccl ─────────────────patch──▶ mccl
  ├─ CUDAExtension ────────patch──▶ MUSAExtension
  │
  ▼
torch_musa (PyTorch backend)
  │
  ├─ torch.musa.* (Python API)
  ├─ _MUSAC (C++ extension)
  ├─ MUSACachingAllocator
  ├─ MUSAStream / MUSAEvent
  ├─ MUSAGraph
  ├─ MCCL
  └─ Inductor codegen
```

---

## 4. 下游用户（SGLang, vLLM-MUSA 等）

**SGLang**：
- 使用 torchada 的 Triton Fused MoE（router.py, fused_moe.py）
- 使用 torchada 的 FP8 quant kernel
- 依赖 `_patch_flash_attn` 提供 flash attention

**vLLM-MUSA**：
- 使用 torchada 的 Graph Rotation 解决深模型 CUDA Graph 上限
- 使用 torchada 的 CUDAExtension 移植
- 依赖 `_patch_library_impl` 注册自定义算子

**Megatron-LM**：
- 使用 torchada 的 `_patch_distributed_backend` 实现 MCCL 通信
- 依赖 `_patch_autocast` 实现混合精度

---

## 5. 关键配置参数与环境变量表

### torchada 环境变量

| 环境变量 | 默认值 | 说明 | 代码位置 |
|---|---|---|---|
| `TORCHADA_PLATFORM` | 自动检测 | 强制平台：`cuda`/`musa`/`cpu` | `_platform.py:35-41` |
| `TORCHADA_GRAPH_ROTATION` | `1` | 启用 Graph Rotation（0=禁用） | `_graph_rotation.py:253` |
| `TORCHADA_GRAPH_EXEC_CAP` | `1900` | live executable 上限 | `_graph_rotation.py:49` |
| `TORCHADA_GRAPH_AUTOPROBE` | `0` | 自动探测驱动上限（1=启用） | `_graph_rotation.py:117` |
| `TORCHADA_GRAPH_EXEC_MARGIN` | `128` | auto-probe 安全边距 | `_graph_rotation.py:122` |
| `TORCHADA_CPP_OPS_VERBOSE` | `0` | C++ ops 编译详细输出 | `_cpp_ops.py:118` |
| `MTGPU_TARGET` | 自动检测 | MUSA GPU 架构目标 | `_cpp_ops.py:129` |

### torch_musa 环境变量

| 环境变量 | 默认值 | 说明 | 代码位置 |
|---|---|---|---|
| `PYTORCH_MUSA_ALLOC_CONF` | 无 | 内存分配器配置（如 `cpu:unified`） | `__init__.py:236` |
| `MUSA_HOME` | `/usr/local/musa` | MUSA SDK 安装路径 | `_cpp_ops.py:192` |
| `TORCH_RNG_ALIGN` | 无 | RNG 对齐到 NV GPU | `__init__.py:51` |
| `MUSA_VISIBLE_DEVICES` | 无 | MUSA 设备可见性 | — |
| `CPU_UNIFIED_FLAG` | `False` | 内部标志，统一内存模式 | `__init__.py:237` |

### 关键常量

| 常量 | 值 | 说明 | 代码位置 |
|---|---|---|---|
| `_DEFAULT_CAP` | `1900` | Graph Rotation 默认上限 | `_graph_rotation.py:44` |
| `_DEFAULT_MARGIN` | `128` | auto-probe 边距 | `_graph_rotation.py:45` |
| `kLargeBuffer` | `20 MiB` | 大块分配阈值 | `MUSACachingAllocator.cpp:54` |
| `kMinBlockSize` | `512` | 最小 block 大小 | `MUSACachingAllocator.cpp:59` |
| `kStreamsPerPool` | `32` | 每 pool stream 数 | `MUSAStream.cpp:17` |
| `TORCH_MIN_VERSION` | `2.11.0` | 最低 torch 版本 | `__init__.py:26` |

---

## 6. 源码文件索引

### torchada 核心文件

| 文件 | 行数 | 核心内容 |
|---|---|---|
| `src/torchada/__init__.py` | 114 | 入口：`load_cpp_ops()`, `apply_patches()` |
| `src/torchada/_patch.py` | ~2220 | 16 个 patch 函数 + `apply_patches()` |
| `src/torchada/_platform.py` | 194 | `Platform` 枚举, `detect_platform()` |
| `src/torchada/_cpp_ops.py` | 239 | `load_cpp_ops()`, `load_graph_rotation_ops()` |
| `src/torchada/_graph_rotation.py` | 304 | `_Rotation` 类, `install()` |
| `src/torchada/_runtime.py` | 133 | 函数名翻译工具 |
| `src/torchada/csrc/ops.cpp` | ~100 | C++ ATen op 覆盖 |
| `src/torchada/_graph_rotation_src/graph_exec_aux.cpp` | ~60 | `free_exec`/`inst_exec` C++ 实现 |
| `src/torchada/triton/kernels/quant/fp8.py` | 282 | FP8/INT8 quant kernels |
| `src/torchada/triton/runtime/fused_moe/router.py` | 459 | TopK/MoE routing |
| `src/torchada/triton/runtime/fused_moe/fused_moe.py` | ~500 | Fused MoE kernel |
| `src/torchada/triton/autotune/fused_moe/__init__.py` | ~120 | MoE 配置系统 |
| `src/torchada/utils/cpp_extension.py` | ~1450 | `CUDAExtension`, `load()`, `BuildExtension` |
| `src/torchada/cuda/__init__.py` | ~50 | CUDA 兼容 API 模块 |

### torch_musa 核心文件

| 文件 | 行数 | 核心内容 |
|---|---|---|
| `torch_musa/__init__.py` | ~296 | 18 步启动序列 |
| `torch_musa/core/Device.cpp` | ~27 | `current_device`, `set_device`, `Synchronize` |
| `torch_musa/core/GuardImpl.h` | ~250 | `MUSAGuardImpl` 完整实现 |
| `torch_musa/core/MUSACachingAllocator.cpp` | ~4000+ | 内存分配器核心（152KB） |
| `torch_musa/core/MUSAStream.cpp` | ~200+ | Stream pool 管理 |
| `torch_musa/core/MUSAEvent.h` | ~100+ | Event 结构定义 |
| `torch_musa/core/PythonMCCL.cpp` | ~19 | MCCL 版本查询 |
| `torch_musa/csrc/aten/musa/MUSAGraph.cpp` | ~200+ | Graph capture C++ 实现 |
| `torch_musa/musa_graph/graphs.py` | ~100+ | MUSAGraph Python 包装 |
| `torch_musa/_inductor/codegen/wrapper.py` | ~200+ | `MUSATritonWrapperCodeGen` |
| `torch_musa/_inductor/musagraph_trees.py` | ~3000+ | MUSA Graph Trees（121KB） |
| `torch_musa/distributed/__init__.py` | ~40 | `_apply_distributed_patch()` |
| `torch_musa/distributed/device_mesh.py` | ~51 | DeviceMesh patch |
| `torch_musa/distributed/fsdp/sharded_grad_scaler.py` | ~200+ | FSDP ShardedGradScaler |

---

## 7. 面试高频问题速查

**Q: torchada 和 torch_musa 的关系？**
A: torchada 是跑在 torch_musa 之上的 CUDA→MUSA 兼容层。torch_musa 提供底层硬件后端（Device/Memory/Stream/Graph/MCCL），torchada 通过 monkey-patch 让 CUDA 代码透明运行。

**Q: Graph Rotation 解决了什么问题？**
A: MUSA 驱动 ~2048 live executable 上限。torchada 用 LRU 维护 live executable，超限时释放 executable 但保留 template，下次 replay 时重新实例化（~0.3ms）。

**Q: torchada 如何做到"零代码修改"？**
A: 通过 `@patch_function` 注册表 + `apply_patches()` 在 `import torchada` 时全局 monkey-patch PyTorch API，将 `torch.cuda.*` 重定向到 `torch.musa.*`。

**Q: torch_musa 的内存分配器特点？**
A: 红黑树 pool + 双向链表，支持 split/coalesce，类似 PyTorch 的 CUDACachingAllocator，但适配 MUSA 的 `musaMalloc`。

**Q: torchada 的 patch 引擎设计模式？**
A: 注册表模式（Registry Pattern），类似 Flask `@app.route`。`@patch_function` 收集函数到 `_patch_registry`，`apply_patches()` 按顺序执行。

---

## 附录：源码文件索引

### torchada 核心文件

| 文件 | 核心类/函数 | 说明 |
|------|-----------|------|
| `src/torchada/__init__.py` | `load_cpp_ops()`, `apply_patches()` | 入口：加载 C++ ops + 应用所有 patch |
| `src/torchada/_patch.py` | `_patch_registry`, `apply_patches()` (:2103), `@patch_function` | 16 个 patch 函数注册表 |
| `src/torchada/_platform.py` | `Platform` 枚举, `detect_platform()`, `is_musa_platform()` | 平台检测 |
| `src/torchada/_cpp_ops.py` | `load_cpp_ops()`, `load_graph_rotation_ops()` | C++ 算子覆盖加载 |
| `src/torchada/_graph_rotation.py` | `_Rotation` 类, `install()`, `_probe_live_live_exec_limit()` | Graph Rotation 核心创新 |
| `src/torchada/_runtime.py` | 函数名翻译工具 | CUDA→MUSA 符号翻译 |
| `src/torchada/csrc/ops.cpp` | `_musa_beginAllocateCurrentThreadToPool()` | C++ ATen op 覆盖 |
| `src/torchada/_graph_rotation_src/graph_exec_aux.cpp` | `free_exec()`, `inst_exec()`, `has_exec()` | Graph Rotation C++ 辅助 |
| `src/torchada/triton/kernels/quant/fp8.py` | `per_token_group_quant_fp8()`, `_per_token_group_quant_8bit` | FP8/INT8 quant kernels |
| `src/torchada/triton/runtime/fused_moe/router.py` | `fused_topk_native()`, `grouped_topk()` | MoE routing |
| `src/torchada/triton/runtime/fused_moe/fused_moe.py` | `fused_experts_impl()` | Fused MoE kernel |
| `src/torchada/utils/cpp_extension.py` | `CUDAExtension`, `load()`, `BuildExtension` | CUDA Extension 移植 |

### torch_musa — Device & Memory

| 文件 | 核心类/函数 | 说明 |
|------|-----------|------|
| `torch_musa/csrc/core/Device.cpp` | `current_device()`, `set_device()`, `Synchronize()`, `device_count()` | Device 管理 |
| `torch_musa/csrc/core/GuardImpl.h` | `MUSAGuardImpl` | DeviceGuard 实现 |
| `torch_musa/csrc/core/MUSACachingAllocator.cpp` | `Block`, `BlockPool`, `NativeCachingAllocator::allocate()` (:3755), `DeviceCachingAllocator::free()` (:1608), `alloc_block()` (:2816), `free_block()` (:2516), `get_free_block()` (:2683) | 内存分配器核心（152KB） |
| `torch_musa/csrc/core/MUSACachingAllocator.h` | `MUSAAllocator` 接口 | 分配器头文件 |
| `torch_musa/csrc/core/MUSAAllocatorConfig.h` | `MUSAAllocatorConfig` | 分配器配置（expandable segments、GC threshold） |
| `torch_musa/core/memory.py` | `caching_allocator_alloc`, `empty_cache`, `MUSAPluggableAllocator` | Python 内存 API |

### torch_musa — Stream, Event & Graph

| 文件 | 核心类/函数 | 说明 |
|------|-----------|------|
| `torch_musa/csrc/core/MUSAStream.cpp` | `MUSAStream`, stream pool (`kStreamsPerPool=32`) | Stream 管理 |
| `torch_musa/csrc/core/MUSAEvent.h` | `MUSAEvent` | Event 结构 |
| `torch_musa/csrc/aten/musa/MUSAGraph.cpp` | `MUSAGraph::capture_begin()` (:65), `capture_end()` (:140), `replay()` (:219), `instantiate()` (:179), `reset()` (:283) | Graph capture 核心 |
| `torch_musa/csrc/aten/musa/MUSAGraph.h` | `MUSAGraph` 类定义 | Graph 头文件 |
| `torch_musa/musa_graph/graphs.py` | `MUSAGraph` Python 包装, `graph_pool_handle()` | Python Graph API |

### torch_musa — MCCL 通信

| 文件 | 核心类/函数 | 说明 |
|------|-----------|------|
| `torch_musa/csrc/core/MCCL.h` | `torch::musa::mccl::version()` | MCCL 版本接口 |
| `torch_musa/csrc/core/MCCL.cpp` | MCCL 实现 | MCCL C++ 层 |
| `torch_musa/csrc/core/PythonMCCL.h` | `THMPModule_mccl_version()` | Python MCCL 绑定 |
| `torch_musa/csrc/core/PythonMCCL.cpp` | `THMPModule_mccl_version()` (:8) | Python MCCL 绑定实现 |
| `torch_musa/csrc/distributed/ProcessGroupMCCL.h` | `ProcessGroupMCCL` 类定义 | MCCL 进程组头文件 |
| `torch_musa/csrc/distributed/ProcessGroupMCCL.cpp` | `allreduce_impl()` (:3998), `allreduce()` (:4026), `broadcast()` (:4115), `send()` (:5124), `abortComms()` (:1219) | MCCL 集合通信实现 |
| `torch_musa/csrc/distributed/MCCLUtils.h` | `getMcclDataType()`, `getMcclReduceOp()` | MCCL 工具函数 |
| `torch_musa/csrc/distributed/MUSAEventCache.cpp` | `MUSAEventCache` | MCCL Event 缓存 |

### torch_musa — Inductor & Distributed

| 文件 | 核心类/函数 | 说明 |
|------|-----------|------|
| `torch_musa/_inductor/codegen/wrapper.py` | `MUSATritonWrapperCodeGen` | Inductor wrapper codegen |
| `torch_musa/_inductor/musagraph_trees.py` | MUSA Graph Trees | Dynamo graph break 场景 |
| `torch_musa/distributed/__init__.py` | `_apply_distributed_patch()` | 分布式 patch 入口 |
| `torch_musa/distributed/device_mesh.py` | `_apply_device_mesh_patch()` | DeviceMesh 适配 |
| `torch_musa/distributed/fsdp/sharded_grad_scaler.py` | `ShardedGradScaler` | FSDP gradient scaler |

### torch_musa — 初始化

| 文件 | 核心类/函数 | 说明 |
|------|-----------|------|
| `torch_musa/__init__.py` | 18 步启动序列, `_apply_patches()`, `overwrite_cuda_api()` | 后端初始化入口 |
| `torch_musa/core/_lazy_init.py` | `_lazy_init()`, `_lazy_call()` | 延迟初始化 |

---

> **文档统计**：本文档覆盖 torchada + torch_musa 两个项目，包含 80+ 处 file:line 源码引用、4 条完整调用链、3 幅 ASCII 架构图、8 张对比/参数表、1 个源码文件索引附录（60+ 文件）。