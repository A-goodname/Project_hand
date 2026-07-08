# Active Context — Project_hand

## 当前焦点

**P4 Y 后退 CRP 实现** — v7 设计已定稿: `P4.XYZ = P3.XYZ + Y后退`, `P4.ABC = P1.ABC`，倒水为纯 B 轴旋转。
RealSense_R 偏移重测定全部完成，融合 Cup 检测已修复。

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

7. **⚠️ 未解决问题**:
   - **P4 Y 后退未实现**: P4 新定义和 Y 偏移量已写入配置和文档, 但机械臂 CRP 端尚未编码. 当前仍使用旧 P4 定义.

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **P4 Y 后退 CRP 实现** | 设计已定, 待机械臂端编码 CposOffset + Y 后退 |
| 🟡 P1 | **HandInCup 无遮挡重测** | 调优 dist_factor/depth_tolerance |
| 🟡 P1 | **HandInCup 有遮挡实测** | 灵巧手遮挡杯把场景 |
| 🟢 P2 | Orchestrator 联调 | 双相机 + 机器人 + 灵巧手全流程 |
