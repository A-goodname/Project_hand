# Active Context — Project_hand

## 当前焦点

**left 模式全流程通过**: P3 示教 + Demo 校正 + Run 实测全部完成。准备 Git 提交。

## 最近变更

1. **left 模式全流程通过** (2026-06-30):
   - left 模式 P3 示教 (3 次取均值, 30°)
   - Demo 中 10°/15°/20° 逐角度实测校正 Y 偏移
   - `p3_teach_offset.left` 四个角度 (10°/15°/20°/30°) 全部填入实测值
   - `estun_handle_cup_run.py` left 模式全流程实测通过
   - **Bug 修复**: PyYAML 整数键 vs `str()` 字符串键不匹配 → 10° 回退问题 → `{str(k): v for k, v in ...}` 统一转字符串键 (demo + run 双文件)

2. **back 模式全流程验证通过** (2026-06-30)

3. **P2 双相机体系 Step 2-4 完成** (2026-06-30): EyeToHand 模块搭建

4. **灵巧手使能可靠性修复** (2026-06-26)

## 下一步

- Git 提交 left + back 模式成果
- P4 预倒水点: 待在实际示教器上测试 (硬件被占用)
- 双相机体系 Step 5-7: Orchestrator 状态机
- `EyeToHand/calibration/calibrate.py` 实操标定 (机械臂被占用)
