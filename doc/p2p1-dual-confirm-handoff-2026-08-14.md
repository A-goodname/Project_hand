# P2/P1 双确认协议适配 — 交接文档（2026-08-14）

> 本文件为会话交接：下一轮在 Project_hand 内的 CRP/协议适配工作以此为准。
> Skill 根目录：`.claude/skills/estun-omnihand-robotics/`（SKILL.md 含全部 CRP 修改规则，动手前必读）。

## 1. 任务来源（对方两份请求，均已存 `~/VLA-BD/docs/`）

1. **D0 执行门禁**（`project-hand-d0-execution-gates-request-2026-08-14.md`，基线 VLA-BD `b8c0b30`）
2. **P2/P1 双确认协议**（`project-hand-zip-controller-barrier-request-2026-08-14.md`）

对方已离线审计 `Project_hand-132051c.bundle`（标定 `calibration_result.yaml` 合格：Daniilidis/26 样本/0.228px）。

## 2. 对方 D0 不可变要求

1. **标定失败必须阻止启动**：所有给 ESTUN 发目标位姿的入口统一走严格共享加载器；文件缺失/YAML 重复键/无 `calibration.handeye.use_method`/方法不存在/非有限刚体 4×4/单位变换/质量不达标 → 记录原因并启动失败。**禁 `np.eye(4)`、默认 Tsai、直接发相机系位姿**。已审计：15 个 py 文件含 np.eye(4) 回退（state_machine.py、estun_handle_sm.py、estun_handle_cup_sm.py、estun_handle_data_sm.py、estun_handle_cup_run.py、estun_greeting_handle_cup_run.py 等）。
2. **PREGRIP 不是位姿确认**：手势指令与位姿确认必须分开。
3. **两个独立一次性确认点**：
   ```
   稳定初始观测 → CONFIRM_PREGRASP_POSE → ESTUN 移至 P2
   → REOBSERVE_HANDLE（腕部 D435i 重观测）
   → CONFIRM_GRASP_POSE → ESTUN 才可移至 P1 并闭合
   ```
   确认结果绑定 `run_id`、单调序号、`ticket_id`、revision、短有效期、单次消费。拒绝/超时/断连/格式错 → 受控停止，**不得复用旧 P1**。

## 3. 对方要求的最小消息序列（仅 Phase A-C 把手边界）

```text
PC 发送预抓取计划（绑定 pregrasp ticket）
CRP: MovL H_P2
CRP → PC: P2_ARRIVED:<pregrasp-ticket-id>
CRP: 等待 GRASP_AUTH:<grasp-ticket-id>
PC: P2 后用腕部 D435i 重观测；确认到位+新观测/标定/租约/目标一致才发 GRASP_AUTH
CRP: MovL H_P1
CRP → PC: P1_ARRIVED:<grasp-ticket-id>
CRP: 等待 GRIP_AUTH:<grasp-ticket-id>
PC: 消费一次性 GRASP ticket 后发 GRIP_AUTH
CRP → PC: GRIPDONE:<grasp-ticket-id>
PC: 仅在有效 GRIPDONE 时闭合灵巧手并记录回执
```

任一等待超时/格式错/ticket 不一致/PC 断连 → CRP 停留/安全停止，不得自动继续。

**禁止改动**：video0 迎宾检测与挥手/复位、杯子递交及 HandInCup 分支、现有速度、P4 倒水和回退路径（除非新等待边界必需）。

## 4. 待回传材料（对方等这批材料后才出精确补丁）

1. CRP 文本版源码（Phase A-C 执行的那个，bundle 里只有 `doc/estun_handle_cup_code_orchestrator.txt` 不够）— 需示教器导出
2. `132051c` 起的 git diff（源码/requirements/非私密配置）— 本机 `git diff 132051c..HEAD` 可生成
3. 运行时实际入口文件名 + 所用 CRP 文件名/版本
4. 每文件 SHA-256 + 基准 commit

**现阶段限制**：不导入控制器、不启机器人/灵巧手/相机，仅离线材料。

## 5. 本机环境事实（已核实）

- 运行入口：`Orchestrator/main.py`（4 阶段调度）；协议实现在 `Orchestrator/state_machine.py`（SharedContext + GreetingHandler + HandleGraspAndPourHandler）
- 最新 CRP：`doc/handlepluscup-orchestrator.crp`（80 widgets, Phase 0+A-D）——当前 `H_P2→H_P1→GRIPDONE`，需适配
- TCP：机器人连本机 `192.168.101.96:8080`（DHCP 动态 IP）；Phase 0 用 8600
- 相机：Camera B 固定 = USB 口 4.2 的 UGREEN（`EyeToHand/config.yaml` `camera_path: usb-0:4.2`）；D435i 在 2e:00.3 独立控制器；**禁硬编码 /dev/videoN，按名称枚举**
- 串口：灵巧手 by-id 路径（460800）；当前会话未重登，需 `sg dialout -c` 包裹
- git：本地提交不 push；身份 `qaq <huangzhiwuming@163.com>`；`core.autocrlf input`；`.gitattributes` 已固化

## 6. CRP 修改要点（skill 提炼）

- `.crp` = 单行 JSON（50K+ tokens），`Read` 打不开 → 用 3 步法分析（顶层 points/widgets → movl 速度 → 定位目标 block）
- **必须 `open(f, 'r', encoding='utf-8')`**（Windows GBK 默认会炸）
- 先备份 `.bak`（gitignore 已配）；用 `scripts/estun_crp_helpers.py` 的 clone_block/new_id
- 陷阱：`data` 字段双态（string/dict，socketclose 必须 dict）；CposOffset pointa/pointb 交换会报"参数错误"；TranStrToCpos 信号位姿不可全零
- 验证：`python scripts/estun_crp_helpers.py <file.crp> --limit 40` + `python -m py_compile`

## 7. 下一步建议

1. 生成 `git diff 132051c..HEAD`（限源码/requirements/配置）→ 回传
2. 示教器导出 CRP 文本版 → 回传
3. 等对方基于实际 CRP 出离线补丁 → 本机跑故障注入回放 → 才轮到 D1 实机
