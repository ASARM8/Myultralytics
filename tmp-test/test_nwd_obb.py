"""
NWD-OBB 功能验证脚本
验证 nwd_obb() 函数的正确性和数值稳定性
"""
import sys
import torch

# 添加项目路径
sys.path.insert(0, r"/root/Myultralytics")

from ultralytics.utils.metrics import nwd_obb

def test_identical_boxes():
    """测试 1：完全相同的 OBB → NWD 应接近 1.0"""
    boxes = torch.tensor([
        [100.0, 200.0, 11.0, 50.0, 0.5],
        [300.0, 400.0, 11.0, 80.0, 1.0],
    ])
    result = nwd_obb(boxes, boxes)
    print(f"[测试 1] 相同框 NWD: {result}")
    assert (result > 0.99).all(), f"相同框的 NWD 应接近 1.0，实际为 {result}"
    print("  ✓ 通过")

def test_distant_boxes():
    """测试 2：相距很远的 OBB → NWD 应接近 0.0"""
    box1 = torch.tensor([[0.0, 0.0, 11.0, 50.0, 0.0]])
    box2 = torch.tensor([[1000.0, 1000.0, 11.0, 50.0, 0.0]])
    result = nwd_obb(box1, box2)
    print(f"[测试 2] 远距离框 NWD: {result}")
    assert (result < 0.01).all(), f"远距离框的 NWD 应接近 0.0，实际为 {result}"
    print("  ✓ 通过")

def test_output_shape():
    """测试 3：验证输出形状 — N 个框对 → N 个标量"""
    N = 100
    box1 = torch.randn(N, 5).abs()
    box1[:, 2:4] = box1[:, 2:4] * 30 + 5  # 合理的 w, h
    box2 = torch.randn(N, 5).abs()
    box2[:, 2:4] = box2[:, 2:4] * 30 + 5
    result = nwd_obb(box1, box2)
    print(f"[测试 3] 输出形状: {result.shape}，期望 ({N},)")
    assert result.shape == (N,), f"输出形状应为 ({N},)，实际为 {result.shape}"
    assert (result >= 0).all() and (result <= 1).all(), f"NWD 值域应在 [0,1]"
    print("  ✓ 通过")

def test_numerical_stability():
    """测试 4：数值稳定性 — 极小宽高不应产生 NaN/Inf"""
    box1 = torch.tensor([[100.0, 100.0, 0.001, 0.001, 0.0]])
    box2 = torch.tensor([[100.0, 100.0, 0.001, 0.001, 0.0]])
    result = nwd_obb(box1, box2)
    print(f"[测试 4] 极小框 NWD: {result}")
    assert torch.isfinite(result).all(), f"结果包含 NaN 或 Inf"
    print("  ✓ 通过")

    # 极大值
    box3 = torch.tensor([[5000.0, 5000.0, 1000.0, 2000.0, 1.5]])
    box4 = torch.tensor([[5000.0, 5000.0, 1000.0, 2000.0, 1.5]])
    result2 = nwd_obb(box3, box4)
    print(f"         极大框 NWD: {result2}")
    assert torch.isfinite(result2).all(), f"结果包含 NaN 或 Inf"
    print("  ✓ 通过")

def test_monotonicity():
    """测试 5：单调性 — 距离越近 NWD 越高"""
    anchor = torch.tensor([[100.0, 100.0, 11.0, 50.0, 0.0]])
    near = torch.tensor([[105.0, 100.0, 11.0, 50.0, 0.0]])
    far = torch.tensor([[150.0, 100.0, 11.0, 50.0, 0.0]])
    
    nwd_near = nwd_obb(anchor, near)
    nwd_far = nwd_obb(anchor, far)
    print(f"[测试 5] 近距离 NWD={nwd_near.item():.4f}，远距离 NWD={nwd_far.item():.4f}")
    assert nwd_near > nwd_far, f"近距离的 NWD 应大于远距离"
    print("  ✓ 通过")

def test_import_from_tal():
    """测试 6：验证 tal.py 中的导入正常"""
    from ultralytics.utils.tal import RotatedTaskAlignedAssigner
    assigner = RotatedTaskAlignedAssigner()
    print(f"[测试 6] RotatedTaskAlignedAssigner 创建成功")
    print(f"         DOBB_EPSILON={assigner.DOBB_EPSILON}, NWD_C={assigner.NWD_C}")
    
    # 测试 iou_calculation 方法
    gt = torch.tensor([[100.0, 100.0, 30.0, 80.0, 0.5]])
    pd = torch.tensor([[102.0, 101.0, 25.0, 75.0, 0.5]])
    result = assigner.iou_calculation(gt, pd)
    print(f"         iou_calculation 输出: {result}")
    assert torch.isfinite(result).all() and (result >= 0).all()
    print("  ✓ 通过")

if __name__ == "__main__":
    print("=" * 60)
    print("  NWD-OBB 功能验证")
    print("=" * 60)
    
    tests = [
        test_identical_boxes,
        test_distant_boxes,
        test_output_shape,
        test_numerical_stability,
        test_monotonicity,
        test_import_from_tal,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"  结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    sys.exit(1 if failed else 0)
