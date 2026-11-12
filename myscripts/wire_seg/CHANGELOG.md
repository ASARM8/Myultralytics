# 电线实例分割实验变更记录

## 2026-11-12 · 四组提示的真值邻线对评估

在已有划分上运行 `bash myscripts/run_wire_seg_server.sh neighbor-pairs val` 或将 `val` 换成 `test`。输入为该划分的缓存和同一运行目录下 target、points、boxes、axes 的四个 mask.pt。候选匹配使用粗框，四组掩膜预测均使用精修框、各候选自己的裁剪及全部邻候选。全程只做推理。

同图块中，任一电线的真值 OBB 条带（短边两侧各加 12 像素，长边两端各加 4 像素）包含另一根线至少一个有效独占像素，即计入一个无序真值对。对两根均有独立匹配且各自掩膜 IoU 达到 0.50 或 0.75 的成功对，分别报告占全部真值对、占两根均已匹配真值对的比例。另列方向差不超过 15° 的子集，以及 axes 和各对照单独成功的数量。报告存放在 `/workspace/workdir/wire_seg/neighbor_pairs_<split>.json`，明细中缺失匹配的编号与 IoU 记为 null，已匹配但无命中记为零。

| 文件 | 用途 |
| --- | --- |
| myscripts/wire_seg/neighbor_pairs.py | 定义真值邻线对、运行四个冻结分割头、输出逐对 IoU 和统一分母的汇总。 |
| myscripts/wire_seg/data.py | 允许匹配时跳过完全位于忽略区的标注并保留原实例编号。 |
| train_wire_seg.py、myscripts/run_wire_seg_server.sh | 提供通用命令行与服务器入口。 |
| tests/test_wire_instance_seg.py | 验证漏检、对称去重、忽略区、阈值边界以及四权重推理与报告写入。 |
| myscripts/wire_seg/CHANGELOG.md | 登记版本用途、评价口径与输出位置。 |

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
