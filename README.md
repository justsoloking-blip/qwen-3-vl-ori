# Qwen3-VL-32B 学生原始工程

从 Qwen3-VL-32B-Instruct 在 SGLang 上的原始部署开始，逐步完成分布式推理问题排查、Profiler 分析和性能优化。本仓库仅提供起步源码、部署工具、请求与采样脚本，以及四次课的实验任务。

**原始含义：未应用本项目的兼容性修复、手写融合算子或新增优化配置。** SGLang 官方版本内置的 CUDA Graph、通信实现和局部编译机制保持原样。开启新的并行配置时，学生可能遇到原工程的问题，这正是第二次课要自行定位的内容。

仓库从全新 Git 历史开始，学习任务中不附补丁、答案、优化结果或完成版 PPT。

## 1. 起点与环境

| 项目 | 原始起点 |
| --- | --- |
| 模型 | Qwen/Qwen3-VL-32B-Instruct，BF16 |
| 框架 | SGLang 0.5.6.post2 |
| Python / PyTorch | 3.12 / 2.9.1+cu128 |
| 默认 GPU / TP | GPU 0，TP1，单张 H200 |
| 上下文 / 并发上限 | 32768 / 8 |
| 显存比例 / Graph 最大 batch | 0.85 / 8 |
| 双卡实验建议 | 两张 H200，记录实际 NVLink / PCIe 拓扑 |

权重、tokenizer、processor 与配置来自[Qwen 官方固定版本](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/0cfaf48183f594c314753d30a4c4974bc75f3ccb)。本仓库不分发权重；模型文件约 66.7 GB，建议另预留充足磁盘用于依赖、缓存和 trace。

源码来自 [SGLang v0.5.6.post2](https://github.com/sgl-project/sglang/tree/v0.5.6.post2)，两个关键模块与该版本官方文件逐字节一致。完整安装依赖快照和安装限制见 [environment/README.md](environment/README.md)。

## 2. 获取与部署

以下命令在独立的 Linux GPU 机器、仓库根目录执行。

```bash
git clone https://github.com/justsoloking-blip/qwen-3-vl-ori.git
cd qwen-3-vl-ori

# 检查原始文件。学生开始修改后，校验不通过属于预期。
python3 scripts/verify_baseline.py --upstream

# 创建全新环境；不会复用已有 .venv。
bash scripts/setup_environment.sh

# 下载固定 revision 的完整模型，包括全部权重分片与 tokenizer/processor。
.venv/bin/python scripts/download_model.py --local-dir models/Qwen3-VL-32B-Instruct

# 只准备 baseline，不包含选择修复版/优化版的入口，也不会自动启动服务。
.venv/bin/python scripts/prepare_runtime.py --model-path models/Qwen3-VL-32B-Instruct
.venv/bin/python runtime/baseline/service.py start
.venv/bin/python runtime/baseline/service.py status
```

权重加载和 warmup 是异步的；观察 `runtime/baseline/logs/server.log`，API ready 后再执行请求。随机 API key 保存在 `runtime/baseline/api.key`，工具自动读取，不要提交该文件。

```bash
.venv/bin/python tests/api_smoke.py
.venv/bin/python benchmarks/opt_probe.py --label baseline --rounds 3
.venv/bin/python runtime/baseline/service.py stop
```

原始服务保留图文与流式接口。`api_smoke.py` 检查接口、鉴权、简单文字和合成图片；不代替完整模型质量评估。

`--gpus 0,1` 仅改变可见 GPU，**不会自动把 TP1 改成 TP2**。并行配置由第二次课自行设置。端口可通过准备脚本的 `--port` 更改。不要在同一个已有 runtime 中反复覆盖实验，使用新的 `--runtime-root` 留存每一步。

## 3. 采集与分析 Profiler

```bash
# Profiler 会影响延迟；性能 benchmark 与 trace 采集应分开执行。
.venv/bin/python benchmarks/opt_probe.py --label baseline --profile --server-profile-dir "$PWD/runtime/baseline/results/traces"
.venv/bin/python benchmarks/analyze_traces.py --root runtime/baseline/results
```

默认在服务所在机器执行采样，输出位于 `runtime/baseline/results/`。远程 API 的 Profiler 输出路径属于**服务器文件系统**；路径设置及参数见 `--help`，完成采集后再将 trace 拷贝到分析机器。

分析工具输出 `trace-summary.json` 与 `kernels.csv`，供学生整理各 rank 的 GPU 活动区间、算子耗时、调用次数和有相关性证据的 graph replay。核函数名称分类是启发式结果，需结合时间线复核。kernel 耗时之和不等于端到端耗时；无 GPU 活动比例不等于 SM 利用率，也不能直接归因为某一种空泡。

## 4. 四次课与提交要求

| 课程 | 学生要完成的工作 | 任务入口 |
| --- | --- | --- |
| 第 1 次：部署与基线 | 模型结构、环境记录、图文服务、固定负载指标 | [01_baseline.md](exercises/01_baseline.md) |
| 第 2 次：分布式排查 | 对照不同并行配置，收集失败证据，自行定位与修复 | [02_distributed.md](exercises/02_distributed.md) |
| 第 3 次：Profiler | 采集各 rank trace，统计热点与时间线空白，提出优先级 | [03_profiler.md](exercises/03_profiler.md) |
| 第 4 次：优化与决策 | 从测量发现候选方案，逐步实现并记录采用/回退结论 | [04_optimization.md](exercises/04_optimization.md) |

每次课在自己的分支提交：配置、代码差异、运行命令、环境、日志及结论。大型 trace 和权重另行保存，提交路径与校验值。任务文件提供交付要求，不附参考实现或标准答案。

## 5. 工程目录与修改方式

```text
baseline/                原始 config、service、两个关键源码模块与来源校验
environment/             历史依赖快照与环境说明
scripts/                 下载、准备 runtime、原始文件校验
benchmarks/              固定输出请求、Profiler 采集和 trace 统计
tests/                   中性 API smoke
exercises/               四次课任务，无答案
runtime/                 本机生成的完整 SGLang overlay、日志与密钥（忽略）
```

安装包提供 SGLang 的其余模块，`prepare_runtime.py` 将完整包复制到独立 runtime，再覆盖 `baseline/src/` 中的源码。因此可在 runtime overlay 中阅读完整引擎。

修改模型源码时，优先编辑 `baseline/src/sglang/` 中的对应文件，然后准备到新的 runtime。其他模块可先从 overlay 复制回相同的 `baseline/src/sglang/` 路径，再纳入自己的 Git 提交。修改运行配置可编辑 `baseline/config.json`；不要把本机权重路径和密钥写入提交。保留初始提交，逐步提交每个实验，便于回退。

## 6. 校验范围

打包时核对了两个原始源码与官方标签的一致性，并在无 GPU 环境执行源码/脚本语法、CLI 和合成数据/模拟接口检查。它们验证打包与测量工具，不构成真实模型推理或性能复测。

原工程的环境快照被保留；本次未在全新 GPU 机器重新下载并安装整套依赖或执行模型。学生应按第一课记录实际安装与 GPU 请求结果，不能将本地工具检查写成 GPU 复现通过。

运行时只做路径、端口、overlay 和密钥的便携适配，具体列表见 `baseline/provenance.json`；原始源码文件本身不被准备脚本修改。
