# Active Context — Project_hand

## 当前焦点

**Orchestrator 联调准备** — P4 Y 后退 CRP 已完成 (handlepluscup-orchestrator.crp)。
RealSense_R 偏移重测定 + Cup 检测修复 + P4 Y 后退全部完成，进入双相机全流程联调阶段。

## 最近变更

1. **RealSense_R Cup 偏移测定** (2026-07-07):
   - `t_cup_offset_left`: RealSense `[0.049, -0.070, 0.100]` → RealSense_R `[0.040, -0.060, 0.080]` (161033 final)
   - `t_cup_offset` (back): RealSense `[0.057, -0.010, 0.139]` → RealSense_R `[0.048, -0.009, 0.098]` (163920 final)
   - 旧值已注释保留，标注相机来源
   - Git 历史确认: cup offset 无历史 fine_tune 记录 (只有 `41525ff` 初始值)

2. **RealSense_R Handle 偏移测定** (2026-07-07):
   - `t_handle_offset`: RealSense `[0.06, -0.028, 0.094]` → RealSense_R `[0.048, -0.037, 0.074]` (164610)
   - Git 历史对比: 历史 RealSense_R 值为 `[0.04, -0.03, 0.080]` (f2e3579 之前)
   - 新旧偏差: X+8mm, Y-7mm, Z-6mm，方向量级合理
   - 待更多实测验证

3. **Demo/Run 切换至 RealSense_R** (2026-07-07):
   - `estun_handle_cup_demo.py`: RealSense/ → RealSense_R/ + 拷贝全部5个相机参数
   - `estun_cup_demo.py`: RealSense/ → RealSense_R/ + 同步 Cup P1 clamp 守卫和防跳变
   - `estun_handle_demo.py`: fine_tune 反算改为 Cup 式增量法
   - `estun_handle_cup_run.py`: RealSense/ → RealSense_R/ + 拷贝全部5个相机参数
   - `estun_config.yaml`: 所有偏移值更新 + 旧值注释保留 + `handle_detection.fine_tune_mode: true`

4. **融合 Demo/Run Cup 检测修复** (2026-07-08):
   - **根因**: `PoseStabilizer.__init__` 中 `_sf = -1`，全局 `_gf = 0`. 首次 `is_stable()` 返回 True 后，`is_generation_valid()` 检查 `-1 != 0` → False → `reset()` 清空 buffer → `last_known_pose` 永远不会被赋值.
   - **修复**: stabilizer 创建后立即调用 `capture_generation()` 同步代际 (`_sf = _gf = 0`). 仅 2 行代码.
   - Demo 不受影响因为其 PoseStabilizer 无代际计数器.
   - ✅ 已修复，待实测验证.

5. **Handle P3 示教验证** (2026-07-08):
   - ✅ `p3_teach_offset` 在 RealSense_R 下验证通过，无需重测.

6. **Handle offset 多次验证** (2026-07-08):
   - ✅ `t_handle_offset = [0.048, -0.037, 0.074]` 多次实测确认稳定.

7. **P4 Y 后退 CRP 实现** (2026-07-08):
   - ✅ 基于 598 (P4 Y后退) + Phase D (HandInCup) 合并生成 `handlepluscup-orchestrator.crp`
   - 2 个新 DCPOS (Retreat1/Retreat2) 复用字段存阈值+Y后退量
   - Cascading If 实现 4 档角度映射: 10°→200mm, 15°→175mm, 20°→150mm, 30°→100mm
   - P4 = [P3.x, P3.y-Y_RETREAT, P3.z, P1.a, P1.b, P1.c]; P4→P3 纯 B 旋转 + Y 前进

8. **HandInCup 无遮挡实测** (2026-07-08):
   - ✅ 实时 Camera B 测试通过: 手腕稳定检测, 2D+深度判定正确
   - 参数调整: `dist_factor` 0.8→1.2, `depth_tolerance` 0.05→0.08 (实测标定)
   - 模型: YOLOv8n-Pose (轻量, 满足实时性)
   - `required_stable`: 5 帧确认
   - 新增功能: 图片模式 (支持文件/文件夹), 诊断模式 (实时+图片, 输出日志)
   - 所有参数集中到 `EyeToHand/config.yaml` → `hand_in_cup` 段

9. **⚠️ 未解决问题**:
   - **EyeToHand 标定未实操**: scale/bias 为默认占位值
   - **递送参数未填入**: `delivery.Y_TEACH/Z_TEACH/ABC` 需示教

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **Orchestrator 联调** | 双相机 + 机器人 + 灵巧手全流程 |
| 🔴 P0 | **CRP 录入示教器** | 导入 handlepluscup-orchestrator.crp + 示教新增点位 |
| 🟡 P1 | **HandInCup 有遮挡实测** | 灵巧手遮挡杯把场景 (无遮挡已通过) |
| 🟡 P1 | **EyeToHand 标定** | 实操 calibrate.py, 填入 scale/bias |
| 🟢 P2 | 填入递送参数 | delivery.Y_TEACH/Z_TEACH/ABC 示教 |
