# Training Glossary — 训练术语表

> 本术语表是 training-crucible 的规范语言参考，确保 knowledge / precision / performance / tickets
> 四个模块使用一致的术语和定义。每个术语以「**英文** (中文) — 一句定义。相关: […]」格式列出。
> 英文缩写首次出现时给出全称，后续可直接使用缩写。新增术语先在本表补充，再在模块中使用。

---

## 快速索引

| 类别 | 锚点 |
|------|------|
| 并行计算 | [1. 并行计算](#1-并行计算-parallel-computing) |
| 训练阶段 | [2. 训练阶段](#2-训练阶段-training-stages) |
| 精度与数值 | [3. 精度与数值](#3-精度与数值-precision--numerics) |
| 性能优化 | [4. 性能优化](#4-性能优化-performance-optimization) |
| 推理 | [5. 推理](#5-推理-inference) |
| 硬件 | [6. 硬件](#6-硬件-hardware) |
| 框架与工具 | [7. 框架与工具](#7-框架与工具-frameworks--tools) |

---

## 1. 并行计算 (Parallel Computing)

**TP** (Tensor Parallelism, 张量并行) — 将权重矩阵按列或按行切分到多卡，每卡计算部分结果后通过 AllReduce 聚合。相关: [AllReduce] [PP]。
**PP** (Pipeline Parallelism, 流水线并行) — 将模型层按阶段切分到多卡，微批次以流水线方式穿过各阶段，存在 pipeline bubble 开销。相关: [Micro-batch] [1F1B]。
**DP** (Data Parallelism, 数据并行) — 每卡持有完整模型副本，数据分片分配到多卡，梯度通过 AllReduce 同步。相关: [AllReduce] [ZeRO] [FSDP]。
**CP** (Context Parallelism, 上下文并行) — 将长序列沿序列维度切分到多卡，每卡处理一段上下文，用于突破单卡序列长度限制。相关: [AllGather]。
**EP** (Expert Parallelism, 专家并行) — 将 MoE 层中的不同专家分配到不同卡，通过 All-to-All 通信在专家间路由 token。相关: [All-to-All]。
**FSDP** (Fully Sharded Data Parallelism, 全分片数据并行) — 将参数、梯度、优化器状态全分片到多卡，需要时通过 AllGather 重建，显著降低单卡内存。相关: [ZeRO-3] [AllGather]。
**ZeRO** (Zero Redundancy Optimizer, 零冗余优化器) — DeepSpeed 提出的消除数据并行冗余存储的技术：ZeRO-1 切分优化器状态、ZeRO-2 切分梯度、ZeRO-3 切分参数。相关: [FSDP] [DP]。
**AllReduce** (全归约) — 集合通信原语，对所有卡数据执行归约（如求和）并将结果广播到每卡，用于梯度同步。相关: [DP] [TP]。
**AllGather** (全收集) — 集合通信原语，每卡贡献一块数据，最终每卡拥有所有卡的完整拼接结果。相关: [FSDP] [TP]。
**ReduceScatter** (归约分散) — 集合通信原语，对所有卡数据归约后将结果切片分散到各卡，常与 AllGather 配对使用。相关: [FSDP] [ZeRO]。
**All-to-All** (全交换) — 集合通信原语，每卡向其他所有卡发送不同数据并从所有卡接收数据，是 MoE 专家路由的核心通信模式。相关: [EP]。
**Broadcast** (广播) — 集合通信原语，将一个卡的数据复制到通信组内所有其他卡。相关: [模型参数同步]。
**Micro-batch** (微批次) — 在流水线并行中，将一个全局批次拆分为多个更小的批次依次注入 pipeline，提升流水线利用率。相关: [PP] [Gradient Accumulation]。
**Gradient Accumulation** (梯度累积) — 在多个微批次上累积梯度后再执行一次参数更新，用更多计算换取等效大 batch size 而不增加显存。相关: [Micro-batch] [DP]。
**1F1B** (One-Forward-One-Backward, 一前一后) — 流水线并行调度策略，稳定阶段每卡交替执行一次前向和一次反向，均衡各阶段显存占用。相关: [PP] [Interleaved Schedule]。
**Interleaved Schedule** (交错调度) — 流水线并行变体，每个 pipeline 阶段持有多个非连续虚拟层（virtual stages），进一步缩小 bubble 比例。相关: [PP] [1F1B]。
**DualPipe** (双管道) — 双向流水线调度，同时从 pipeline 两端注入微批次，使前向和反向的通信可重叠，提升互联带宽利用率。相关: [PP]。

---

## 2. 训练阶段 (Training Stages)

**Pre-training** (预训练) — 在海量无标注数据上以自回归或掩码目标训练模型，学习通用表征能力，是模型能力的基础阶段。相关: [Post-training] [MFU]。
**Post-training** (后训练) — 在预训练基座上进行的后续训练，包括 SFT、DPO、RLHF 等对齐技术，使模型满足人类偏好和任务要求。相关: [SFT] [RLHF] [DPO]。
**SFT** (Supervised Fine-Tuning, 监督微调) — 在有标注指令-回答数据上对预训练模型进行监督学习，教会模型遵循指令格式。相关: [Post-training] [DPO]。
**DPO** (Direct Preference Optimization, 直接偏好优化) — 利用偏好对数据直接优化策略模型，无需显式训练奖励模型，通过对比 chosen/rejected 样本学习人类偏好。相关: [RLHF] [SFT]。
**RLHF** (Reinforcement Learning from Human Feedback, 基于人类反馈的强化学习) — 先训练奖励模型拟合人类偏好，再用 RL 算法（如 PPO）优化策略模型。相关: [PPO] [Reward Model] [Reference Model]。
**GRPO** (Group Relative Policy Optimization, 群组相对策略优化) — 通过同题多采样的组内相对奖励估计优势函数，无需独立的价值模型，是开源 RL 训练的主流算法。相关: [PPO] [Rollout]。
**PPO** (Proximal Policy Optimization, 近端策略优化) — 经典策略梯度 RL 算法，通过裁剪概率比限制策略更新幅度，保证训练稳定性。相关: [RLHF] [Value Model] [Policy Model]。
**Rollout** (采样生成) — RL 训练中由策略模型生成回答样本的过程，是连接训练与推理的关键环节，生成质量直接影响梯度估计。相关: [GRPO] [On-policy]。
**On-policy** (在策略) — 训练数据由当前正在训练的策略模型实时生成，保证梯度估计无偏，但要求频繁同步权重到推理引擎。相关: [Off-policy] [Rollout]。
**Off-policy** (离策略) — 训练数据可由历史或外部策略生成，数据复用效率高但引入分布偏差。相关: [On-policy]。
**Reward Model** (奖励模型) — 对模型输出打分以拟合人类偏好的模型，在 RLHF 中作为环境奖励的来源。相关: [RLHF] [PPO]。
**Policy Model** (策略模型) — RL 中正在被优化的目标模型，其参数通过最大化期望奖励来更新。相关: [PPO] [GRPO] [Reference Model]。
**Value Model** (价值模型) — 在 PPO 中估计状态价值函数，用于计算优势函数（Advantage），GRPO 通过组内相对奖励替代此模型。相关: [PPO] [GRPO]。
**Reference Model** (参考模型) — RL 训练中冻结的初始策略副本，用于计算 KL 散度惩罚防止策略偏离太远。相关: [RLHF] [DPO] [PPO]。

---

## 3. 精度与数值 (Precision & Numerics)

**Loss NaN** (损失非数) — 训练损失变为 NaN（Not a Number），通常由数值溢出、除零、log(0) 等异常运算导致，标志训练已失效。相关: [Loss Spike] [Numerical Overflow]。
**Loss Spike** (损失尖峰) — 训练损失出现突然的异常跳升，可能由脏数据、学习率过大、数值不稳定等引起，需排查后决定是否回退 checkpoint。相关: [Loss NaN] [Grad Norm]。
**Loss Divergence** (损失发散) — 训练损失持续上升不收敛，通常由学习率配置错误、数据问题或精度设置不当导致。相关: [Loss Spike] [Epsilon]。
**Grad Norm** (梯度范数) — 梯度张量的 L2 范数，用于衡量梯度大小；训练中常监控其变化判断收敛状态和数值稳定性。相关: [Gradient Clipping] [Loss Spike]。
**Gradient Clipping** (梯度裁剪) — 当梯度范数超过阈值时按比例缩小梯度，防止梯度爆炸导致训练不稳定。相关: [Grad Norm] [Loss Spike]。
**Mixed Precision** (混合精度) — 训练中同时使用高精度（FP32）和低精度（FP16/BF16），前向和反向用低精度加速，权重更新用高精度保持稳定。相关: [FP16] [BF16] [Loss Scaling]。
**FP16** (半精度浮点, 16-bit Floating Point) — IEEE 754 半精度格式，动态范围较小，训练中易溢出，需配合 Loss Scaling 使用。相关: [BF16] [Mixed Precision]。
**BF16** (Brain Float 16, 脑浮点 16) — Google 提出的 16 位格式，与 FP32 同动态范围但精度更低，训练中比 FP16 更稳定，常免 Loss Scaling。相关: [FP16] [Mixed Precision]。
**FP8** (8-bit Floating Point, 8 位浮点) — 8 位浮点格式（E4M3 / E5M2），用于进一步加速计算和通信，对缩放因子敏感，是新一代训练和推理的主流低精度格式。相关: [FP16] [Loss Scaling]。
**FP4** (4-bit Floating Point, 4 位浮点) — 4 位浮点格式，用于极限压缩场景（如权重存储），精度损失大，需配合量化感知技术。相关: [Quantization] [INT4]。
**INT8** (8 位整数) — 8 位整数量化格式，常用于推理加速和训练中的通信压缩，动态范围有限需校准。相关: [Quantization] [FP8]。
**INT4** (4 位整数) — 4 位整数量化格式，用于权重压缩（如 GPTQ-4bit），显著降低显存但引入量化误差。相关: [Quantization] [GPTQ] [AWQ]。
**Loss Scaling** (损失缩放) — 在混合精度训练中将损失乘以一个缩放因子，避免反向传播时小梯度在低精度下下溢为零。相关: [Mixed Precision] [FP16] [FP8]。
**Epsilon** (数值稳定常数) — 在除法、开方、归一化等运算中加入的极小常数（如 1e-5、1e-6），防止除零和数值不稳定。相关: [Loss Divergence]。
**Numerical Overflow** (数值上溢) — 计算结果超出数据类型能表示的最大值，在 FP16 中尤为常见，可导致 Inf 和 NaN。相关: [Loss NaN] [FP16]。
**Numerical Underflow** (数值下溢) — 计算结果小于数据类型能表示的最小正值，在梯度计算中常见，导致有效信息丢失为零。相关: [Loss Scaling] [FP16]。

---

## 4. 性能优化 (Performance Optimization)

**Throughput** (吞吐量) — 单位时间内处理的样本数或 token 数（samples/s 或 tokens/s），是衡量训练系统效率的核心指标。相关: [MFU] [TFLOPS]。
**TFLOPS** (Tera FLoating-point Operations Per Second, 万亿次浮点运算每秒) — 硬件或系统每秒执行的浮点运算次数，衡量计算吞吐的硬件级指标。相关: [MFU] [Throughput]。
**MFU** (Model FLOPs Utilization, 模型 FLOPs 利用率) — 实际达到的 FLOPs 占硬件理论峰值 FLOPs 的百分比，衡量训练对硬件计算能力的利用效率。相关: [TFLOPS] [Throughput]。
**Activation Recompute** (激活重计算) — 前向时不保存中间激活值，反向时重新计算以节省显存，用计算时间换内存空间。相关: [Gradient Checkpointing] [CPU Offload]。
**Gradient Checkpointing** (梯度检查点) — 与 Activation Recompute 同义，选择部分层保存检查点（checkpoint），其余层反向时重算，是显存优化的标准技术。相关: [Activation Recompute]。
**CPU Offload** (CPU 卸载) — 将优化器状态、参数或激活值卸载到 CPU 内存，突破 GPU/NPU 显存瓶颈，代价是增加设备间传输。相关: [ZeRO] [Activation Recompute]。
**KV Cache** (键值缓存) — 推理时缓存已计算的 Key 和 Value 注意力状态，避免重复计算历史 token，是推理加速的核心机制。相关: [PagedAttention] [Prefix Caching]。
**PagedAttention** (分页注意力) — 将 KV Cache 按固定大小分页管理，支持动态序列长度和内存共享，解决显存碎片问题。相关: [KV Cache] [vLLM]。
**CUDA Graph** (CUDA 图) — NVIDIA 的图模式执行技术，将一系列 kernel 录制为图后整体回放，消除 kernel launch 开销。相关: [NPU Graph]。
**NPU Graph** (NPU 图模式) — 华为昇腾的图模式执行技术，类似 CUDA Graph，将算子子图编译后整体执行，降低 host-device 调度开销。相关: [CUDA Graph] [CANN]。
**Speculative Decoding** (投机解码) — 用小模型（草稿模型）快速生成候选 token，再由大模型验证，加速自回归推理。相关: [Draft Model] [Acceptance Rate]。
**Continuous Batching** (连续批处理) — 推理服务中动态将新请求插入正在执行的批次，避免等待整批完成，提升 GPU 利用率。相关: [Throughput]。
**Sequence Packing** (序列打包) — 将多条短序列拼接至固定长度送入训练，消除 padding 浪费，提升训练效率。相关: [Throughput] [Pre-training]。

---

## 5. 推理 (Inference)

**Quantization** (量化) — 将模型权重或激活从高精度压缩到低精度表示（如 FP8、INT4），降低显存占用和通信量，分权重量化、激活量化、KV Cache 量化。相关: [INT4] [FP8] [GPTQ] [AWQ]。
**W4A16** (4-bit Weight / 16-bit Activation) — 权重 4 位量化、激活保持 16 位的混合量化配置，是推理部署的常见平衡点。相关: [Quantization] [GPTQ] [AWQ]。
**W8A8** (8-bit Weight / 8-bit Activation) — 权重和激活都 8 位量化的对称配置，精度损失小，是推理和训练推理混合场景的主流选择。相关: [Quantization] [FP8]。
**GPTQ** (GPT Quantization) — 基于一阶近似（Hessian）的权重量化方法，逐列校准量化参数，常用于 4 位推理部署。相关: [Quantization] [W4A16]。
**AWQ** (Activation-aware Weight Quantization, 激活感知权重量化) — 按激活值幅度缩放权重通道以保护重要通道的量化方法，量化精度通常优于 GPTQ。相关: [Quantization] [W4A16]。
**KV Cache Quantization** (KV 缓存量化) — 对推理时的 KV Cache 进行低精度量化（如 FP8、INT8），降低长序列推理的显存占用。相关: [KV Cache] [Quantization]。
**Prefix Caching** (前缀缓存) — 推理服务中缓存相同前缀（如 system prompt）的 KV Cache，命中时跳过重复计算，降低首 token 延迟。相关: [KV Cache] [PagedAttention]。
**Draft Model** (草稿模型) — 投机解码中用于快速生成候选 token 的小模型，其生成速度远快于目标大模型。相关: [Speculative Decoding] [Acceptance Rate]。
**Acceptance Rate** (接受率) — 投机解码中被大模型验证通过的草稿 token 比例，决定实际加速比，受草稿模型与目标模型分布相似度影响。相关: [Speculative Decoding] [Draft Model]。

---

## 6. 硬件 (Hardware)

**Ascend** (昇腾) — 华为自研 AI 处理器品牌，面向训练和推理提供 NPU 算力，代表产品昇腾 910B/910C。相关: [CANN] [HCCL] [Atlas]。
**Atlas** (Atlas 服务器) — 华为基于昇腾 NPU 构建的服务器/模组产品集群，面向数据中心 AI 训练和推理场景。相关: [Ascend] [HCCL]。
**HCCL** (Huawei Collective Communication Library, 华为集合通信库) — 华为昇腾的集合通信库，功能对标 NVIDIA NCCL，为 NPU 间通信提供 AllReduce、AllGather 等原语。相关: [NCCL] [Ascend]。
**NCCL** (NVIDIA Collective Communications Library, NVIDIA 集合通信库) — NVIDIA GPU 的集合通信库，提供 AllReduce、AllGather、Broadcast 等原语，是分布式训练的通信基础。相关: [HCCL] [NVLink]。
**NVLink** (NVLink 高速互联) — NVIDIA GPU 间的高速直连互联技术，带宽远高于 PCIe，是节点内多卡通信的首选通道。相关: [NCCL] [RoCE]。
**RoCE** (RDMA over Converged Ethernet, 融合以太网远程直接内存访问) — 在以太网上实现 RDMA 的高速网络技术，用于跨节点 GPU/NPU 通信，成本低于 InfiniBand。相关: [NVLink] [NCCL]。
**HCCS** (Huawei Cache Coherence System, 华为缓存一致性系统) — 华为昇腾芯片间的高速互联总线，提供缓存一致性支持，用于节点内多 NPU 直连通信。相关: [Ascend] [GMI]。
**GMI** (Global Memory Interconnect, 全局内存互联) — 华为昇腾的全局内存互联协议，提供高带宽内存访问和芯片间数据共享能力。相关: [HCCS] [Ascend]。
---

## 7. 框架与工具 (Frameworks & Tools)

**Megatron-LM** (Megatron-LM) — NVIDIA 推出的分布式大模型训练框架，以多维并行（TP/PP/EP/CP）和高效 transformer 实现著称。相关: [TP] [PP] [EP]。
**DeepSpeed** (DeepSpeed) — 微软提出的分布式训练优化库，以 ZeRO 系列显存优化和 Zero-Offload 闻名，常与 Megatron 组合使用。相关: [ZeRO] [FSDP]。
**MindSpeed** (MindSpeed) — 华为昇腾生态的训练框架，面向 NPU 优化分布式并行和通信策略，是 CANN 生态的训练核心。相关: [Ascend] [HCCL]。
**torchtitan** (torchtitan) — Meta 推出的 PyTorch 原生大模型训练框架，以 FSDP2 和原生并行实现为特色，支持预训练和 TitanRL。相关: [FSDP] [PP] [CP]。
**torch_npu** (torch_npu) — 华为昇腾的 PyTorch 设备适配插件，使 PyTorch 能在 NPU 上运行，是昇腾 AI 生态的基础适配层。相关: [Ascend] [CANN]。
**op-plugin** (op-plugin) — 华为昇腾的算子插件库，提供 PyTorch 算子的 NPU 实现和图模式融合优化。相关: [torch_npu] [CANN]。
**miles** (miles) — 面向 GRPO/PPO 的开源 RL 训练框架，强调 true-on-policy 训练和 SGLang 推理集成。相关: [GRPO] [Rollout] [On-policy]。
**slime** (slime) — 面向 GRPO/PPO 的开源 RL 训练框架，支持 agentic RL 和 SGLang 推理后端，注重可观测性和可复现性。相关: [GRPO] [Rollout]。
**SGLang** (SGLang) — 面向大模型推理和服务的框架，提供 RadixAttention 前缀缓存和高性能后端，是 RL rollout 的常用引擎。相关: [vLLM] [Prefix Caching] [Rollout]。
**vLLM** (vLLM) — 加州大学伯克利开源的大模型推理服务引擎，以 PagedAttention 和 Continuous Batching 为核心特性。相关: [SGLang] [KV Cache] [PagedAttention]。
**CANN** (Compute Architecture for Neural Networks, 神经网络计算架构) — 华为昇腾的统一计算架构，包含算子库、图引擎和运行时，是昇腾 NPU 的软件基座。相关: [Ascend] [ACL] [ATB]。
**ACL** (Ascend Computing Language, 昇腾计算语言) — 华为昇腾的底层编程接口，提供算子开发和设备管理 API，是 CANN 的开发基础层。相关: [CANN] [Ascend]。
**ATB** (Ascend Tensor Boost, 昇腾张量加速库) — 华为昇腾的张量加速库，提供高性能 attention 和 transformer 算子实现。相关: [CANN] [Ascend]。

---

> **使用约定**：其他模块引用术语时以此表定义为准；如需新增术语，先在本表补充定义，再在对应模块中使用。
