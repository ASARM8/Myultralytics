# 电线实例分割实验变更记录

## 2026-11-10 · CA 粗框与精修框的配对分割评估

分支为 `innovation2`，父提交为 `e3e2956a2`。本次修订通过 amend 替换首版 `b8c3bd20d`。可用 `git log --oneline -- myscripts/wire_seg/CHANGELOG.md` 查询修订后的提交号。

**实验口径**　两组使用同一份在精修框上训练的 `axes` 权重、相同的候选身份与检测分数。以 CA 粗框确定真值匹配、邻线子集、切换对象和切换评分区域。在相同电线上比较两组的掩膜 AP、目标覆盖、邻线泄漏及实例切换结果，并记录框 IoU 和掩膜 IoU 的逐实例变化。

**修复记录**　首版在写入配对报告时，对 NumPy 布尔值求和产生了 `numpy.int64`，导致标准 JSON 编码器在输出 `iou_change` 时抛出 `TypeError`。修订版从框与掩膜 IoU 变化交叉表按列求和，得到 Python 整数。计数字段依然表示掩膜 IoU 改善、持平和退化的实例数。新增回归用例通过实际写入和读回 JSON 校验三类计数及空子集。

| 文件（相对工程根目录） | 本次提交中的用途 |
| --- | --- |
| `train_wire_seg.py` | 注册 `paired` 命令，接收评估数据、权重、匹配阈值和输出路径。 |
| `myscripts/run_wire_seg_server.sh` | 提供服务器 `paired val` 和 `paired test` 入口，复用已有权重与候选缓存。 |
| `myscripts/wire_seg/learning.py` | 供常规评估和配对评估共用同一套权重加载与单裁剪掩膜推理。 |
| `myscripts/wire_seg/paired.py` | 校验一一对应的候选框，执行两组推理，生成配对汇总和逐实例明细。本次修复位于汇总计数处。 |
| `tests/test_wire_instance_seg.py` | 覆盖固定邻线、未匹配候选、两侧输入相同、完整推理及 NumPy 差值的 JSON 写入与读回。 |
| `myscripts/wire_seg/CHANGELOG.md` | 记录版本沿革、实验比较口径、修复原因和文件用途。 |

在服务器的 `/workspace/code` 下执行 `bash myscripts/run_wire_seg_server.sh smoke`，通过后运行 `bash myscripts/run_wire_seg_server.sh paired val`。验证集报告保存在 `/workspace/workdir/wire_seg/axes/paired_geometry_val.json`。
