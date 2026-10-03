# 环境边界

`requirements.lock.txt` 是原工程的完整 pip-freeze 快照，关键版本见 `baseline.json`。它用于保留历史条件，不能等同于已经在任何新机器验证通过的安装配方。

安装脚本先安装 CUDA 12.8 对应的 PyTorch wheel，再以 `--no-deps` 安装快照，以保留历史 cuDNN 9.16 覆盖。该覆盖与部分包声明的版本约束存在差异；`pip check` 可能报告冲突。请保留实际安装日志和 `pip check` 输出，在完成 import、GPU 服务和请求验证后记录环境。不要把冲突输出隐藏或当作 GPU 已验证。

```bash
nvidia-smi
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze
.venv/bin/python -c 'import torch,sglang,transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__)'
```

建议在独立 Linux 环境中操作。原始配置只使用单张 H200；第二次课的双卡实验另需两张 GPU 及实际通信拓扑。课程负载和显存需求随上下文、图像分辨率及并发变化，不保证在较小显存卡上直接适用。

