# 仓库指南

## 项目结构与研究主线
主包位于 `ultralytics/`，核心改动集中在 `ultralytics/utils/tal.py`、`ultralytics/utils/loss.py`、`ultralytics/nn/modules/head.py` 和 `ultralytics/cfg/models/11/`。测试在 `tests/`，训练脚本在根目录，诊断脚本在 `myscripts/`，研究记录在 `mydocs/创新点一/`，实验结果和图表在 `res/`、`t/`、`work-dirs/`。

`mydocs/创新点一/` 的结论已经收敛：当前主线是 `Coverage-Aware Assignment + reg_max=32 + fully-decoupled Refine Head`，最终路线优先对应 `yolo11-obb-ca.yaml` 与 `yolo11-obb-ca-refine.yaml`。NWD 修正版主要用于训练稳定性解释；StripPooling、SDA、WAG、弱耦合 refine、`short-side-only` refine 等属于诊断或对照材料，除非任务明确要求，不要把它们改写成新的主线。

## 构建、测试与开发命令
安装开发环境：

```powershell
pip install -e .
pip install -e ".[dev]"
```

常用验证命令：

```powershell
pytest
pytest tests/test_python.py
pytest --cov=ultralytics --cov-report=term-missing
```

实验脚本运行前必须先核对数据集、权重、输出目录和 YAML 路径，例如 `python train_yolo11_obb_refine.py`、`python train_yolo11_obb_refine.py --mode val_ab --weights <ckpt>`、`python myscripts/check_h1h2_stats.py`。

## 实验与评估约定
创新点一的主要指标优先看 `mAP50-95`，不要只用 `mAP50` 解释收益。涉及 CA 后续结构改动时，必须同时对比 baseline、CA、`CA-continue`，避免把继续训练红利误判为模块收益。涉及 Refine Head 时，默认解释口径是 `coarse-only`；`normal refine` 只能作为 A/B 对照，用来判断推理侧 refine 是否污染指标。

## 代码风格与命名规范
使用 Python 3.8+，行宽保持 120 字符以内。函数、变量、脚本文件使用 `snake_case`，类名使用 `PascalCase`，公共 API 使用 Google 风格 docstring。项目配置了 Ruff、isort、YAPF 和 docformatter；可用时运行 `ruff format .` 与 `ruff check .`。新增几何逻辑应使用 PyTorch 张量操作，避免在大规模 anchor/GT 矩阵上写 Python 循环。

## 测试指南
测试框架为 pytest，并默认启用 doctest。新增覆盖优先放进现有 `tests/test_*.py` 文件。修改 TAL、OBB loss、DFL、Refine 或 YAML 解析后，至少运行相关单测和一次小规模脚本级验证；Coverage-Aware 改动还应重新检查 H1/H2：DFL 溢出率是否下降，长目标是否更多分配到 P4/P5。

## 提交、文档与安全
提交信息保持简短中文说明或版本式实验说明，例如 `v8.5.3 ...`。涉及实验结论的改动，要同步更新 `mydocs/创新点一/` 中对应工作记录或总报告。不要提交私有数据集、凭据、大型权重文件或 `/root/autodl-tmp` 等机器绑定路径；可复用代码应通过配置和参数传入路径。
