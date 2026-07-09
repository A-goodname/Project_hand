# Active Context — Project_hand

## 当前焦点

**打招呼 Phase 0 + CRP 全部重写完成** — PC 端和 CRP 均已重写为 4 步单 Socket 持久连接协议. 待联调验证.

## 最近变更

1. **打招呼 Phase 0 + CRP 重写 (未提交)** (2026-07-09):
   - PC 端: `Phase0Server` 类替代旧的 `run_phase0()` 函数
     - 4 步协议 (单连接): 初始位姿→ZHAOSHOU1→YOLO人检测→ZHAOSHOU2→挥手→WOSHOU→握手→Home
     - YOLOv8n 人检测 (Phase 0 独立 Camera pipeline), 挥手 (do_gesture → PAPER), 握手 (PAPER→压力→HAND_SHAKE→松手→PAPER)
     - Phase 0 运行在主线程, 独立 Camera pipeline 在人检测后立即关闭, 不与 Phase A-C 共用
   - CRP: 30 个新 widget 替代旧的 60 个
     - 单 SocketCreate(8600) → 4 步 Send/Read → SocketClose
     - 全部 MovL, 复用现有点位变量
   - Phase A-C (Handle+Cup) 完整保留, `_handle_client_impl` + `handle_client` 与 `estun_handle_cup_run.py` 完全一致
   - 流程文档: `doc/Greeting_Flow.md`

2. **EyeToHand 标定废弃** (2026-07-08):
   - Cup P3 XYZ/ABC 全示教固定
   - calibrate.py 重写: 3D→1D 人体标定 (SVD, person_3d_cam→X_base)

3. **Omnihand 手势+压力检测** (2026-07-08):
   - Gesture: HAND_SHAKE(19)/HELLO_1(20)/HELLO_2(21)
   - 压力检测: get_pressure/wait_for_pressure/wait_for_pressure_release

4. **双相机对调** (2026-07-08):
   - Camera A: RealSense_R (手动像素映射)
   - Camera B: RealSense (rs.align)

## 未解决问题

- **打招呼联调待验证**: CRP + PC 全流程实测
- **HandInCup 有遮挡未实测**
- **EyeToHand 标定暂废弃**: 递送全示教
- **递送参数未填入**: `delivery.X/Y/Z/A/B/C_TEACH` 需示教

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **打招呼联调** | PC 端 + 新 CRP 全流程实测 |
| 🔴 P0 | **HandInCup 有遮挡实测** | 灵巧手遮挡杯把+手腕 |
| 🟡 P1 | **EyeToHand 标定** | 3D→1D 人体标定实操 |
| 🟢 P2 | 递送参数示教 | delivery 全字段 |
