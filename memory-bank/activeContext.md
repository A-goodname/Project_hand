# Active Context — Project_hand

## 当前焦点

**back 模式全流程跑通**: Handle+Cup 抓取+倒水全部通过实测。准备 left 模式示教与实际测试。

## 最近变更

1. **back 模式全流程验证通过** (2026-06-30):
   - RealSense 断连后重连 → `estun_handle_cup_run.py` 崩溃 (`_poses_cleared` UnboundLocalError) → 修复 `main()` global 声明
   - B 角策略回退: `f2e3579` 防跳变 89/91 → `867552d` clamp [80,90]→85 → 最终收紧为 [84,86]→85, 保留 `ORIENTATION_MODE == "vertical"` 守卫
   - `pack_three` + 机械臂脚本: `()` → `<>` (机器人 `TranStrToCpos` 不支持 `()`)
   - 移除逐帧 `\r` 打印, 新增 `[Handle STABLE]` / `[Cup STABLE]` 稳定位姿打印
   - P3 多倒水角度支持: `estun_config.yaml` 按角度索引 (10°/15°/20°/30°), `pour_angle` 配置项切换

2. **P2 双相机体系 Step 2-4 完成** (2026-06-30): EyeToHand 模块搭建

3. **灵巧手使能可靠性修复** (2026-06-26)

4. **通信协议与流程重构** (2026-06-25)

## 下一步

- **left 模式示教**: `t_cup_offset_left` 实际标定 + P3 倒水点示教
- left 模式 P3 多角度配置 (10°/15°/20°/30°) 填入 YAML
- left 模式全流程实测
- Git 提交当前 back 模式成果
- P4 预倒水点: 待在实际示教器上测试 (硬件被占用)
- 双相机体系 Step 5-7: Orchestrator 状态机
