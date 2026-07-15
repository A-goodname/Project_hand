# Active Context — Project_hand

## 当前焦点

**Orchestrator 重构完成 — 状态机 → 4 阶段顺序调度. CONTINUE/ABORT 双分支均测试通过 ✅**

## 最近变更

### v12 编排重构 + 联调修复 (2026-07-15)

1. **状态机 → 4 阶段调度**:
   - 删除 StateHandler/StateMachine/IdleHandler/OrchestratorState
   - 主循环改为 `greeting → handle_detect → seated_check → hand_in_cup` 顺序阶段
   - 任何时刻只有一个检测器运行: greeting(Camera B椅子标定,竞争不大) → handle_detect(Camera A, Camera B暂停) → seated_check(Camera B恢复) → hand_in_cup(Camera B)
   - 删除全部跳帧补丁 (`_pd_skip`, `_hic_skip`, `skip_frames` 保留但保留检测跳帧)

2. **SharedContext 简化**:
   - 删除 `run_seated_detection`, `hand_in_cup_active`, `handle_cup_complete`
   - 新增 `greeting_done` / `targets_sent` / `ready_received` / `hand_judged` 四个 Event

3. **CRP CposOffset 修复**:
   - CONTINUE 分支 C_P4 = CposOffset(C_P3, DCp2) 的 pointa/pointb 引用点交换 → "参数错误"
   - 修正: pointa→DCp2(DCPOS), pointb→C_P3(CPOS)

4. **CONTINUE/ABORT 信号修复**:
   - Python 发可达位姿 (Phase A 收到的机器人初始位姿) 替代全零
   - CRP TranStrToCpos 成功解析

5. **`_camera_a_to_close` 修复**: Phase B 完成后保存引用, main 线程关闭

6. **hand_judged 时机修复**: 从 CONTINUE/ABORT 发送后 → daemon 线程最终清理时

7. **HandInCup 循环修复**: `hand_on_cup is None` → `None or not` (False 时继续检测)

8. **白色把手回退删除**: handle_cup_detector 找不到白色把手时不再搜壶身红色

9. **CRP 速度**: 倒水 H_P3↔H_P4 和 ABORT P3↔P4 使用 sudu(慢速)

10. **Skill 更新**: 新增 CposOffset 陷阱 / TranStrToCpos 信号规则 / HandInCup 循环规则 / 4 阶段架构

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🟢 | 完整流程稳定性测试 | 多次连续运行验证 |
| 🟢 | MediaPipe clearcut 彻底抑制 | shutdown 时 ERROR 日志泄漏 |
| 🟢 | HandInCup 诊断日志清理 | 减少终端输出 |
