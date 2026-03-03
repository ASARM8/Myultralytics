"""
拓扑非极大值抑制 (T-NMS) 核心算法实现

根据 step2-1.md 文档实现的 T-NMS 算法，用于将多个首尾相连或部分重叠的
短预测框"缝合"成完整的长框，解决极细长电线检测中的碎片化问题。

核心思路：
    基于三大拓扑条件（角度一致性、法向共线距离、轴向邻近距离）判断两个 OBB
    是否应该融合，然后通过端点对齐生成新的长 OBB。

OBB 表示格式:
    [x, y, w, h, theta, conf, class_id]
    - (x, y): 中心点坐标（像素坐标）
    - w: 短边宽度（像素）
    - h: 长边长度（像素）（约定 h >= w）
    - theta: 长边方向与 X 轴的夹角（弧度，范围 [-pi/2, pi/2) 或 [0, pi)）
    - conf: 置信度
    - class_id: 类别 ID

使用方法：
    # 作为模块导入
    from tnms import topology_nms
    merged = topology_nms(obbs, tau_theta=10, tau_perp=8, tau_gap=20)

    # 命令行测试（使用随机数据）
    python tnms.py --test
"""

import argparse
import math

import numpy as np


# ======================== 默认参数 ========================
DEFAULT_TAU_THETA = 10.0   # 角度容忍阈值（度）
DEFAULT_TAU_PERP = 8.0     # 法向偏移容忍阈值（像素）
DEFAULT_TAU_GAP = 20.0     # 轴向断点间隙容忍（像素）


# ======================== 条件检查函数 ========================

def check_angle_similarity(theta1, theta2, tau_theta_rad):
    """
    条件 1：角度一致性 (Angular Similarity)

    检查两个 OBB 的长边方向是否趋于平行。
    考虑旋转框角度的周期性（180° 翻转），使用最小夹角约束。

    公式：
        Δθ = min(|θ1 - θ2|, π - |θ1 - θ2|) ≤ τ_θ

    Args:
        theta1: 第一个 OBB 的角度（弧度）
        theta2: 第二个 OBB 的角度（弧度）
        tau_theta_rad: 角度容忍阈值（弧度）

    Returns:
        bool: 是否满足角度一致性条件
    """
    diff = abs(theta1 - theta2)
    delta_theta = min(diff, math.pi - diff)
    return delta_theta <= tau_theta_rad


def check_perpendicular_distance(o1, o2, tau_perp):
    """
    条件 2：法向共线距离 (Perpendicular Collinearity)

    检查 O2 的中心点是否落在 O1 所在的直线上（法向距离极小）。
    将 O2 的中心点投影到 O1 的法向量方向上。

    公式：
        D⊥ = |(y2 - y1)·cos(θ1) - (x2 - x1)·sin(θ1)| ≤ τ⊥

    Args:
        o1: 第一个 OBB [x, y, w, h, theta, ...]
        o2: 第二个 OBB [x, y, w, h, theta, ...]
        tau_perp: 法向偏移容忍阈值（像素）

    Returns:
        bool: 是否满足法向共线距离条件
    """
    x1, y1 = o1[0], o1[1]
    x2, y2 = o2[0], o2[1]
    theta1 = o1[4]

    d_perp = abs((y2 - y1) * math.cos(theta1) - (x2 - x1) * math.sin(theta1))
    return d_perp <= tau_perp


def check_axial_proximity(o1, o2, tau_gap):
    """
    条件 3：轴向邻近距离 (Axial Proximity)

    检查两个 OBB 在同一条直线上且距离不太远（必须首尾相连或有重叠）。
    计算两个中心点在 O1 轴向上的投影距离。

    公式：
        D∥ = |(x2 - x1)·cos(θ1) + (y2 - y1)·sin(θ1)|
        D∥ ≤ (h1 + h2) / 2 + τ_gap

    Args:
        o1: 第一个 OBB [x, y, w, h, theta, ...]
        o2: 第二个 OBB [x, y, w, h, theta, ...]
        tau_gap: 允许的最大断点间隙（像素）

    Returns:
        bool: 是否满足轴向邻近距离条件
    """
    x1, y1 = o1[0], o1[1]
    x2, y2 = o2[0], o2[1]
    theta1 = o1[4]
    h1, h2 = o1[3], o2[3]

    d_parallel = abs((x2 - x1) * math.cos(theta1) + (y2 - y1) * math.sin(theta1))
    max_dist = (h1 + h2) / 2.0 + tau_gap

    return d_parallel <= max_dist


def can_merge(o1, o2, tau_theta_rad, tau_perp, tau_gap):
    """
    综合检查两个 OBB 是否满足三大融合条件。

    必须同时满足：
        1. 角度一致性
        2. 法向共线距离
        3. 轴向邻近距离

    Args:
        o1: 第一个 OBB [x, y, w, h, theta, conf, class_id]
        o2: 第二个 OBB [x, y, w, h, theta, conf, class_id]
        tau_theta_rad: 角度容忍阈值（弧度）
        tau_perp: 法向偏移容忍阈值（像素）
        tau_gap: 轴向断点间隙容忍（像素）

    Returns:
        bool: 是否应该融合
    """
    # 前提：必须是同一类别
    if int(o1[6]) != int(o2[6]):
        return False

    # 条件 1：角度一致性
    if not check_angle_similarity(o1[4], o2[4], tau_theta_rad):
        return False

    # 条件 2：法向共线距离
    if not check_perpendicular_distance(o1, o2, tau_perp):
        return False

    # 条件 3：轴向邻近距离
    if not check_axial_proximity(o1, o2, tau_gap):
        return False

    return True


# ======================== 缝合操作 ========================

def get_endpoints(obb):
    """
    根据 OBB 的 (x, y, h, theta) 计算长边方向上的两个端点。

    端点 = 中心点 ± (h/2) * 方向向量

    Args:
        obb: OBB [x, y, w, h, theta, ...]

    Returns:
        (p_a, p_b): 两个端点坐标，各为 (x, y)
    """
    x, y = obb[0], obb[1]
    h = obb[3]
    theta = obb[4]

    # 长边方向的单位向量
    dx = math.cos(theta) * h / 2.0
    dy = math.sin(theta) * h / 2.0

    p_a = (x - dx, y - dy)  # 尾端
    p_b = (x + dx, y + dy)  # 头端

    return p_a, p_b


def merge_obbs(o1, o2):
    """
    缝合操作：将两个满足融合条件的 OBB 合并为一个新的长 OBB。

    步骤（根据 step2-1.md 第3节）：
        1. 提取两个 OBB 的 4 个端点
        2. 找出距离最远的两个端点 → 确定新长度
        3. 取最远端点的中点 → 确定新中心
        4. 用最远端点连线斜率 → 确定新角度
        5. 置信度取最大值，宽度取最大值

    Args:
        o1: 第一个 OBB [x, y, w, h, theta, conf, class_id]
        o2: 第二个 OBB [x, y, w, h, theta, conf, class_id]

    Returns:
        merged: 新的 OBB [x, y, w, h, theta, conf, class_id]
    """
    # 步骤 1：提取 4 个端点
    p1a, p1b = get_endpoints(o1)
    p2a, p2b = get_endpoints(o2)
    endpoints = [p1a, p1b, p2a, p2b]

    # 步骤 2：找距离最远的两个端点
    max_dist = 0.0
    p_start, p_end = endpoints[0], endpoints[1]

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            pi, pj = endpoints[i], endpoints[j]
            dist = math.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2)
            if dist > max_dist:
                max_dist = dist
                p_start, p_end = pi, pj

    # 步骤 2：新长度 h_new = 最远两点的欧氏距离
    h_new = max_dist

    # 步骤 3：新中心 = 最远两端点的中点
    x_new = (p_start[0] + p_end[0]) / 2.0
    y_new = (p_start[1] + p_end[1]) / 2.0

    # 步骤 4：新角度 = 最远端点连线的斜率
    theta_new = math.atan2(p_end[1] - p_start[1], p_end[0] - p_start[0])

    # 步骤 5：置信度和宽度取最大值
    conf_new = max(o1[5], o2[5])
    w_new = max(o1[2], o2[2])

    # 类别保持一致（已经在 can_merge 中检查过）
    class_id = int(o1[6])

    return [x_new, y_new, w_new, h_new, theta_new, conf_new, class_id]


# ======================== T-NMS 主算法 ========================

def topology_nms(obbs, tau_theta=DEFAULT_TAU_THETA, tau_perp=DEFAULT_TAU_PERP,
                 tau_gap=DEFAULT_TAU_GAP, verbose=False):
    """
    拓扑非极大值抑制 (T-NMS) 主算法。

    基于贪心合并的迭代融合过程（参考 step2-1.md 第4节）：
        1. 将所有 OBB 按置信度从高到低排序
        2. 依次弹出种子框，与剩余框尝试融合
        3. 种子框变长后重新遍历（可能吃掉更远的框）
        4. 无法再融合时，将种子框存入结果

    Args:
        obbs: OBB 列表，每个元素为 [x, y, w, h, theta, conf, class_id]
              支持 list 或 numpy array
        tau_theta: 角度容忍阈值（度，会自动转为弧度）
        tau_perp: 法向偏移容忍阈值（像素）
        tau_gap: 轴向断点间隙容忍（像素）
        verbose: 是否打印详细日志

    Returns:
        keep_boxes: 融合后的 OBB 列表
    """
    if len(obbs) == 0:
        return []

    # 角度从度转为弧度
    tau_theta_rad = math.radians(tau_theta)

    # 转为 list of list 方便操作
    if isinstance(obbs, np.ndarray):
        boxes = obbs.tolist()
    else:
        boxes = [list(b) for b in obbs]

    # 步骤 1：按置信度从高到低排序（索引 5 为 conf）
    boxes.sort(key=lambda b: b[5], reverse=True)

    original_count = len(boxes)
    keep_boxes = []
    merge_count = 0

    # 步骤 2-4：贪心合并
    while len(boxes) > 0:
        # 步骤 3a：弹出第一个框作为种子框
        seed = boxes.pop(0)

        # 步骤 3b-3e：迭代尝试融合
        merged_in_this_round = True
        while merged_in_this_round:
            merged_in_this_round = False
            i = 0
            while i < len(boxes):
                target = boxes[i]

                # 检查三大融合条件
                if can_merge(seed, target, tau_theta_rad, tau_perp, tau_gap):
                    # 执行缝合
                    seed = merge_obbs(seed, target)
                    # 从列表中删除该目标框
                    boxes.pop(i)
                    merged_in_this_round = True
                    merge_count += 1

                    if verbose:
                        print(f"  [融合] 种子框吃掉一个框，"
                              f"新长度: {seed[3]:.1f}px, "
                              f"剩余框数: {len(boxes)}")

                    # 关键：重新从头遍历剩余框（种子框变长了，可能吃掉更远的框）
                    i = 0
                else:
                    i += 1

        # 步骤 3c：没有框能再融合时，存入结果
        keep_boxes.append(seed)

    if verbose:
        print(f"\n  T-NMS 完成: {original_count} 个框 → {len(keep_boxes)} 个框 "
              f"(发生 {merge_count} 次融合)")

    return keep_boxes


def topology_nms_batch(all_obbs_dict, tau_theta=DEFAULT_TAU_THETA,
                       tau_perp=DEFAULT_TAU_PERP, tau_gap=DEFAULT_TAU_GAP,
                       verbose=False):
    """
    批量 T-NMS：对多张图的预测结果分别执行 T-NMS。

    Args:
        all_obbs_dict: dict，键为图片名/ID，值为该图的 OBB 列表
        tau_theta: 角度容忍阈值（度）
        tau_perp: 法向偏移容忍阈值（像素）
        tau_gap: 轴向断点间隙容忍（像素）
        verbose: 是否打印详细日志

    Returns:
        merged_dict: dict，键为图片名/ID，值为融合后的 OBB 列表
        stats: 统计信息 dict
    """
    merged_dict = {}
    total_before = 0
    total_after = 0

    for img_name, obbs in all_obbs_dict.items():
        before = len(obbs)
        merged = topology_nms(obbs, tau_theta, tau_perp, tau_gap, verbose=False)
        after = len(merged)

        merged_dict[img_name] = merged
        total_before += before
        total_after += after

        if verbose and before != after:
            print(f"  {img_name}: {before} → {after} 个框")

    stats = {
        "total_images": len(all_obbs_dict),
        "total_boxes_before": total_before,
        "total_boxes_after": total_after,
        "total_merged": total_before - total_after,
        "avg_boxes_before": total_before / max(len(all_obbs_dict), 1),
        "avg_boxes_after": total_after / max(len(all_obbs_dict), 1),
    }

    if verbose:
        print(f"\n  批量 T-NMS 统计:")
        print(f"    总图片数: {stats['total_images']}")
        print(f"    总框数（融合前）: {stats['total_boxes_before']}")
        print(f"    总框数（融合后）: {stats['total_boxes_after']}")
        print(f"    总融合次数: {stats['total_merged']}")

    return merged_dict, stats


# ======================== 工具函数 ========================

def load_obbs_from_txt(txt_file):
    """
    从 txt 文件加载 OBB 数据。

    格式: x y w h theta conf class_id

    Args:
        txt_file: txt 文件路径

    Returns:
        obbs: numpy array, shape (N, 7)
    """
    obbs = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                x, y, w, h, theta, conf = [float(p) for p in parts[:6]]
                class_id = int(parts[6])
                obbs.append([x, y, w, h, theta, conf, class_id])
    return np.array(obbs) if obbs else np.empty((0, 7))


def save_obbs_to_txt(obbs, txt_file):
    """
    将 OBB 数据保存到 txt 文件。

    格式: x y w h theta conf class_id

    Args:
        obbs: OBB 列表或 numpy array
        txt_file: 输出文件路径
    """
    with open(txt_file, "w") as f:
        for obb in obbs:
            x, y, w, h, theta, conf = obb[0], obb[1], obb[2], obb[3], obb[4], obb[5]
            class_id = int(obb[6])
            f.write(f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} {theta:.6f} {conf:.6f} {class_id}\n")


def load_gt_from_txt(txt_file):
    """
    从 txt 文件加载 GT 标注数据。

    格式: x y w h theta class_id

    Args:
        txt_file: txt 文件路径

    Returns:
        gts: numpy array, shape (N, 6)
    """
    gts = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                x, y, w, h, theta = [float(p) for p in parts[:5]]
                class_id = int(parts[5])
                gts.append([x, y, w, h, theta, class_id])
    return np.array(gts) if gts else np.empty((0, 6))


# ======================== 测试与演示 ========================

def run_test():
    """运行一个简单的测试用例，验证 T-NMS 算法的正确性。"""
    print("=" * 60)
    print("  T-NMS 算法测试")
    print("=" * 60)

    # 模拟场景：一根水平电线被分成了 3 段短预测框
    # 电线从 (100, 200) 到 (700, 200)，总长 600 像素
    obbs = [
        # x,    y,   w,   h,     theta, conf, class
        [200.0, 200.0, 5.0, 200.0, 0.0, 0.85, 0],  # 左段：中心(200,200), 长200
        [400.0, 202.0, 5.0, 200.0, 0.02, 0.90, 0],  # 中段：中心(400,202), 稍有偏移
        [600.0, 200.0, 5.0, 200.0, -0.01, 0.80, 0], # 右段：中心(600,200), 稍有角度偏差
        [400.0, 500.0, 5.0, 150.0, 1.57, 0.75, 0],  # 垂直方向的另一根线（不应被融合）
    ]

    print(f"\n  输入: {len(obbs)} 个 OBB")
    for i, obb in enumerate(obbs):
        print(f"    #{i}: 中心=({obb[0]:.0f}, {obb[1]:.0f}), "
              f"宽={obb[2]:.0f}, 长={obb[3]:.0f}, "
              f"角度={math.degrees(obb[4]):.1f}°, "
              f"置信度={obb[5]:.2f}")

    # 执行 T-NMS
    merged = topology_nms(obbs, tau_theta=10, tau_perp=8, tau_gap=20, verbose=True)

    print(f"\n  输出: {len(merged)} 个 OBB")
    for i, obb in enumerate(merged):
        print(f"    #{i}: 中心=({obb[0]:.0f}, {obb[1]:.0f}), "
              f"宽={obb[2]:.0f}, 长={obb[3]:.0f}, "
              f"角度={math.degrees(obb[4]):.1f}°, "
              f"置信度={obb[5]:.2f}")

    # 验证
    assert len(merged) == 2, f"预期 2 个框（水平线融合 + 垂直线保留），实际 {len(merged)} 个"
    # 融合后的水平线应该大约有 600 像素长
    horizontal_box = [b for b in merged if abs(b[4]) < 0.5][0]
    assert horizontal_box[3] > 500, f"融合后水平线长度应 > 500，实际 {horizontal_box[3]:.1f}"

    print("\n  [PASS] 测试通过！三段水平短框成功融合为一条长框，垂直线保持独立。")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T-NMS 拓扑非极大值抑制")
    parser.add_argument("--test", action="store_true", help="运行测试用例")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        print("请使用 --test 参数运行测试用例，或作为模块导入使用。")
        print("示例: python tnms.py --test")
