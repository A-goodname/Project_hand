# Active Context — Project_hand

## 当前焦点

**RealSense_R 偏移重新测定** — 从 RealSense (rs.align) 切换到旧相机 RealSense_R (手动像素映射)，
需要重新 fine_tune 所有灵巧手偏移。Cup back/left 已完成，Handle 初次完成，融合 Demo/Run 待修复。

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

4. **⚠️ 未解决问题**:
   - **融合 Demo/Run Cup 检测失败**: 单独 Cup Demo (`start_pipeline=True`) 正常; 融合模式 (`start_pipeline=False` + inject_frames) cup 无位姿. 已拷贝全部5个相机参数仍未解决. 导致 run 无法完成 stable → 全流程阻塞.
   - **P4 Y 后退未实现**: P4 新定义和 Y 偏移量已写入配置和文档, 但机械臂 CRP 端尚未编码. 当前仍使用旧 P4 定义.

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **修复融合 Demo/Run Cup 检测** | start_pipeline=False 时 cup 无输出, run 全流程阻塞 |
| 🔴 P0 | **P4 Y 后退 CRP 实现** | 设计已定, 待机械臂端编码 CposOffset + Y 后退 |
| 🟡 P1 | **Handle P3 示教验证** | 旧 p3_teach_offset 在 RealSense_R 下是否仍有效 |
| 🟡 P1 | **Handle offset 多测几次** | 对比历史值 `[0.04, -0.03, 0.080]`，确认稳定性 |
