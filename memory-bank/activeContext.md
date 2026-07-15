# Active Context — Project_hand

## 当前焦点

**Orchestrator 联调 — Phase 0 通过, Phase A-D 调试中。CRP Cup 递送段新增 C_P5 防碰撞点。**

## 最近变更

1. **Camera A pipeline 全生命周期共享** (2026-07-15):
   - GREETING._start() 创建 HandleCupDetector (color+depth), Phase 0 人检测复用其 color 帧
   - HANDLE._start() 检测已存在 → 只补稳定器+标定

2. **Camera B 预加载 + 就位后停止** (2026-07-15):
   - 提到 GREETING.on_enter 启动
   - 人就位确认后 `run_seated_detection=False`, 停止 Camera B YOLO, Handle+Cup 独占资源

3. **红色胶带检测重写** (2026-07-15):
   - 对齐 RealSense_R/detect_red_tape: HSV双阈值(S≥50 V≥50) + 开运算 + 最大连通域 + min_area≥20
   - 移除 visionpro.py 的 R优势/Morph close/多级过滤/评分系统

4. **白色把手阈值** (2026-07-15): min_value 110→80, min_rgb 120→90, search_left_frac 0.68→0.80

5. **CRP 修改** (2026-07-15):
   - SocketClose 修正: Socket1→Socket0
   - 删除 Phase0→PhaseA 重复 Wait + C_P3→READY 间多余 Wait
   - 新增 C_P5: CposOffset(C_P1, DCp3), DCp3.dz=100mm, 正常 P1→P5→P3, ABORT P3→P5→P1
   - C_P5 MovL 后补 Wait 1s

6. **Phase B fixed 模式修复** (2026-07-15):
   - compute_grasp_rotation 仅非 fixed 分支调用
   - 稳定器+标定在 ctx.handle_cup_detector 之前创建

7. **TOCTOU 竞态修复** (2026-07-15):
   - Camera A 不在 daemon 线程 close, Pipeline 存活到程序退出 (对齐 estun_greeting)

8. **日志系统** (2026-07-15):
   - Orchestrator/logger.py — 终端+文件双输出
   - 关键事件日志: Phase A/B/C/D, 人就位, HandInCup

9. **修复集** (2026-07-15):
   - 椅子检测: 去掉 has_chair 门控
   - 人就位: leave_frames 5→30, YOLO 漏检容忍
   - HandInCup 超时刷屏: 仅首次打印
   - MediaPipe clearcut: hand_in_cup.py 内 GLOG_minloglevel=3
   - Camera B D435i serial 死代码删除
   - Handle P3 left 模式各角度 Y-2cm (10°/15°/20°/30°)

10. **Skill + TXT 更新** (2026-07-15):
    - Skill: CRP 修改规则重写 (7条实战陷阱), Camera B→UGREEN, HandInCup→MediaPipe
    - TXT: v10 changelog + C_P5 路径 + 多余 Wait 删除

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | Phase C/D 全流程测试 | PREGRIP/GRIPDONE/CUP/READY/RELEASE 完整链路 |
| 🟡 | 人就位检测改进 | bbox 稳定性追踪方案 |
