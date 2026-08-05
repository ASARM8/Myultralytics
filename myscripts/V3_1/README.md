# Refine V3.1 实验目录

## 当前状态

本目录同时保存两类互不覆盖的工作：

1. 已完成的 V3.1 冻结前结构验证；
2. 在原有数据划分上正式实现的 V3.1 训练、复现和等预算选择。

正式 V3.1 固定使用短边+长边残差、全部 proposal、无推理质量门控、无第二次 NMS。训练入口包括 `geometry_only` 和训练期 `quality_aux` 两个等预算版本。完整命令、文件用途和验收规则见 [V31_TRAINING.md](V31_TRAINING.md)。旧 V3 代码和结果不被覆盖。

## 1. 冻结前验证目的

以下冻结前工具负责验证旧 V3 结果支持删除哪些实现细节。它们不训练 V3.1，不读取测试集，也不从多个变体中自由挑最高分。

固定检查如下：

1. 排除 train/val 中已经证实的逐字节重复图像，形成 clean-val 敏感性评估；
2. 在原训练规则对应的 deterministic train-holdout 上复核机制方向；
3. 比较完整双边残差与仅短边残差；
4. 在同一短边残差下比较质量门控与全 proposal；
5. 检查精修后重新执行 NMS 是否真正改变指标；
6. 使用预声明的非劣性与恒等性门槛生成 V3.1 设计建议。

上述检查只决定 V3.1 的结构删减方向。最终证据支持保留短边与长边残差、取消推理门控并删除 re-NMS；修改后的 Head 必须从固定纯 CA 权重重新训练。

## 2. 固定边界

```bash
export OMP_NUM_THREADS=1

BASE=/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt
DATA=/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml
V3_EXPORT=/root/autodl-tmp/paper_exports/refine_v3_seed0_01
V3_RUN=$V3_EXPORT/train_seed0_scale_only
CHECKPOINT=$V3_RUN/checkpoints/best.pt
V31_EXPORT=/root/autodl-tmp/paper_exports/refine_v31_prefreeze_seed0
```

- 纯 CA 路径必须使用上面的 canonical 路径；
- `imgsz` 固定为 640；
- checkpoint 固定为 V3 已在 train-holdout 选择的 epoch15；
- train-holdout 参数必须复现 V3 训练：fraction=0.20、seed=0、`group_regex` 为空；
- FP32、batch=8；
- test 全程封存。

## 3. 代码回归测试

```bash
python -m pytest -q tests/V3 tests/V3_1
```

这些测试检查排除清单、固定分组、候选变体定义、非劣性规则和恒等性规则。它们不替代真实 GPU 数据评估。

## 4. 准备 clean-val 与数据来源审计

```bash
python -m myscripts.V3_1.prepare_prefreeze_v31 \
  --data "$DATA" \
  --split-audit-dir "$V3_EXPORT/dataset_split_audit" \
  --imgsz 640 \
  --batch 8 \
  --workers 8 \
  --holdout-fraction 0.20 \
  --holdout-seed 0 \
  --holdout-group-regex '' \
  --scene-regex '^([^_]+)_' \
  --output-dir "$V31_EXPORT/split_preparation"
```

主要输出：

| 文件 | 含义 |
|---|---|
| `clean_val_exclusions.txt` | clean-val 必须排除的 7 张逐字节重复 val 图像 |
| `clean_holdout_exclusions.txt` | train-fit 与 train-holdout 精确重复时，需要从诊断 holdout 排除的图像 |
| `exact_overlap_label_audit.csv` | 13 对重复图像的标签哈希一致性 |
| `train_fit_holdout_exact_overlap.csv` | 原 V3 内部分组中跨 fit/holdout 的逐字节重复 |
| `train_fit_holdout_source_frame_overlap.csv` | 原 V3 内部分组中跨 fit/holdout 的同源帧切片 |
| `source_frame_overlap.csv` | 去掉切片坐标后，同一源帧是否跨 train/val |
| `scene_overlap_heuristic.csv` | 按文件名前缀推断的场景交叉，仅作命名语义核对 |
| `prioritized_near_duplicate_review.csv` | 优先显示同源帧、同场景的近重复人工复核队列 |
| `prefreeze_split_manifest.json` | 计数、规则和 test 未使用声明 |

该步骤会重新计算 13,559 张 train 图像的 SHA256，以检查 V3 原始 fit/holdout 是否存在内部精确重复，耗时主要取决于存储读取速度。`scene_regex` 只有在文件名前缀确实代表采集场景时才具有场景语义。dHash 近重复仍不能自动判为泄漏。

## 5. train-holdout 机制复核

```bash
python -m myscripts.V3.audit_refine_v3 \
  --checkpoint "$CHECKPOINT" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --evaluation-scope train-holdout \
  --holdout-fraction 0.20 \
  --holdout-seed 0 \
  --group-regex '' \
  --exclude-images-file "$V31_EXPORT/split_preparation/clean_holdout_exclusions.txt" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --skip-baseline-reference \
  --output-dir "$V31_EXPORT/audit_train_holdout"
```

这里跳过的是“与既有 0.45413 聚合分数比较”，因为 train-holdout 没有预先登记的 CA mAP；CA 文件哈希、纯 CA Head、roundtrip 和 gate-off 恒等性检查仍然保留。若 fit/holdout 没有精确重复，排除文件只含注释，脚本会按零排除处理。

## 6. clean-val 机制复核

```bash
python -m myscripts.V3.audit_refine_v3 \
  --checkpoint "$CHECKPOINT" \
  --ca-weights "$BASE" \
  --data "$DATA" \
  --evaluation-scope val \
  --exclude-images-file "$V31_EXPORT/split_preparation/clean_val_exclusions.txt" \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --no-amp \
  --skip-baseline-reference \
  --output-dir "$V31_EXPORT/audit_clean_val"
```

clean-val 预期为 1688 张图像，即原 1695 张减去 7 张唯一精确重复图像。脚本会要求排除清单中的每一张图都能在当前 val 中找到，否则拒绝运行。

## 7. 生成冻结前判定

```bash
python -m myscripts.V3_1.summarize_prefreeze_v31 \
  --holdout-audit-dir "$V31_EXPORT/audit_train_holdout" \
  --clean-val-audit-dir "$V31_EXPORT/audit_clean_val" \
  --split-preparation-dir "$V31_EXPORT/split_preparation" \
  --noninferiority-tolerance 0.002 \
  --identity-tolerance 0.0005 \
  --minimum-refine-gain 0.002 \
  --output-dir "$V31_EXPORT/decision"
```

判定采用固定顺序：

1. `short_only_all` 相对 `all_refine` 在 clean-holdout 和 clean-val 均不劣超过 0.002，才删除长边分支；
2. `short_only_all` 相对 `short_only` 均不劣超过 0.002，才取消推理质量门控；
3. `short_only_all` 与 `short_only_all_no_renms` 的七项指标最大绝对差均不超过 `5e-4`，才删除二次 NMS；
4. `short_only_all` 在两个范围均满足 ΔmAP50-95≥0.002、AP75 不下降、AP90 下降不超过 0.002，才认为 V3.1 主方向可进入重训实现。

主要输出为 `v31_prefreeze_decision.json` 和 `v31_prefreeze_decision.md`。

## 8. 结果返回清单

下载整个目录：

```text
/root/autodl-tmp/paper_exports/refine_v31_prefreeze_seed0
```

必须至少包含：

```text
split_preparation/
audit_train_holdout/
audit_clean_val/
decision/
```

只有分析完这些结果后才实现正式 V3.1 Head。quality head 是否保留为纯辅助监督不能由推理开关决定，届时需要一个等训练预算的单因素重训练对照。
