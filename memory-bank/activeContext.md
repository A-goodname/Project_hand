# Active Context — Project_hand

## 当前焦点

**Orchestrator 状态机模块完成** (Step 5-7 代码完成); 待实操标定后联调。

## 最近变更

1. **Orchestrator 状态机模块创建** (2026-07-02):
   - `Orchestrator/state_machine.py`: 6 状态线性机 (IDLE → HANDLE_GRASP_AND_POUR → LOCATE_PERSON → APPROACH_PERSON → WAIT_AND_RELEASE → RETREAT)
   - `Orchestrator/main.py`: 双相机主循环 (Camera A 眼在手上 + Camera B 眼在手外)
   - 复用了现有 Handle+Cup 的 PoseStabilizer / pack_three / fmt_pose 等关键工具
   - TCP 递送协议: 单 pose `<x,y,z,A,B,C>` + READY 确认 + RELEASE
   - 状态机异常处理: 任何异常回退到 IDLE

2. **双模块 serial_number 参数统一** (2026-07-02):
   - `RealSense_R/realsense_shared.py` + `RealSense/realsense_shared.py`: `BaseRealSenseDetector.__init__` 加 `serial_number=None` 参数
   - `EyeToHand/person_detector.py`: 透传 `serial_number` 给父类
   - 零影响: 现有代码不传此参数时走 else 原路径

3. **P4 预倒水点验证成功** (2026-07-02):
   - 已在示教器上实际测试通过, 倒水流程完整可用

4. **left 模式全流程通过** (2026-06-30)

5. **back 模式全流程验证通过** (2026-06-30)

6. **P2 双相机体系 Step 2-4 完成** (2026-06-30): EyeToHand 模块搭建

## 下一步

- **`EyeToHand/calibration/calibrate.py` 实操标定** — 当前唯一阻塞项 (需机械臂)
- **填入 `EyeToHand/config.yaml` 递送参数**: `delivery.Y_TEACH/Z_TEACH/ABC` 预示教值
- **双相机联调**: Orchestrator 全流程实测 (Step 8-10)
