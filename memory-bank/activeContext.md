# Active Context — Project_hand

## 当前焦点

**Orchestrator 联调 — Phase 0 ✅, Phase A-C ✅, Phase D HandInCup 判定通过 ✅，CONTINUE 路径调试中。**

## 最近变更

### v11 联调修复 (2026-07-15)

1. **Handle P2 可达性修复**:
   - estun_config `pre_offset.t_offset`: `[0.0,-0.10,0.03]` → `[0,0.10,-0.05]`
   - P2 更靠近机器人可达空间，逆解成功

2. **CRP Handle 流程简化 (删 PREGRIP)**:
   - 第一次 P2 后不再发送 PREGRIP：`P2→P1→GRIPDONE` 直接夹紧
   - RESET 从 P2 移到 P1：`P1→RESET→P2→Home`
   - P2 后无灵巧手动作

3. **CRP while 循环优化**:
   - 删除 Phase B/C `SocketReadStr` while 内的 `Wait 500ms`
   - SocketReadStr 持续监听，避免 Python 发包时 CRP 在 Wait 中丢包

4. **CRP 倒水延时**: `Wait 5s` → `Wait 3s`

5. **CRP C_P4 安全退避**: `DCp2.dz = Home.z-C_P1.z` → 固定 `DCp2.dz = 100mm`

6. **CRP CONTINUE IF**: 删除 `equal` 条件判断

7. **HandInCup 检测调通**:
   - YOLO model: yolov8m → **yolov8n** (更轻量)
   - device 加入 config: `hand_in_cup.device: cuda`
   - 超时 5s → **10s** (含 CUDA warmup 首帧 ~380ms)
   - 诊断日志: 打印每步耗时 + cup/hand 状态 + stable 计数
   - 独立测试通过: cup 稳定 ~35ms, stable 10/5 ✅

8. **人就位 + HandInCup 跳帧**:
   - 人就位: 每 3 帧跑一次 (main.py `_pd_skip = 3`)
   - HandInCup: 每 2 帧跑一次 (main.py `_hic_skip = 2`)
   - Camera B 分辨率已配置 1280×720

9. **MediaPipe clearcut**: 增强 env var (`GLOG_stderrthreshold=4`, `GRPC_VERBOSITY=NONE`)，shutdown 仍有残留

10. **Skill 更新**: data-as-string 陷阱 — socketclose inside if blocks 必须 dict

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | CONTINUE 路径全流程测试 | C_P4→RESET→Home→SocketClose 完整链路 |
| 🟡 | ABORT 路径测试 | P3→P5→P1→RELEASE→P2→Home |
| 🟡 | MediaPipe clearcut 彻底抑制 | shutdown 时 ERROR 级日志仍有泄漏 |
| 🟢 | HandInCup 诊断日志清理 | 测试稳定后可降级或移除 |
