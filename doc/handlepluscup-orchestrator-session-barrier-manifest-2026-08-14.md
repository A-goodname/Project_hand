# handlepluscup-orchestrator-session-barrier 离线模板 manifest（2026-08-14）

对应工单：`project-hand-estun-session-bound-barrier-work-order-2026-08-14.md`
阶段：D0，仅离线编辑与回传。本次未连接/使能/运行任何硬件。

## 1. 文件与版本

| 项 | 值 |
|----|-----|
| 来源文件 | `doc/handlepluscup-orchestrator.crp`（锁定源） |
| 来源 SHA-256 | `342cb0e971773d9b113a5bdd0a108b36b1bb17fa9cdf225b8fa5adc218f5bd3a` |
| 输出文件 | `doc/handlepluscup-orchestrator-session-barrier-2026-08-14.crp` |
| 输出 SHA-256 | `3d16aa59fa64fd6b6bf1985ed1306af5d827c9f8d3b0b09f4304646a4af810bf` |
| 输出大小 | 207262 字节 |
| 编辑器版本 | **v1.6.3b**（Codroid/ESTUN 编辑器，与创建来源 CRP 相同环境） |
| 控制器/固件版本 | `unknown`（本机无示教器可读） |
| 生成方式 | 从来源工程**克隆同类型 block**（完整保留 type/data/id/parents/index/style），仅修改参数与变量/点引用，新块 id 按编辑器 id 格式生成；未发明任何新 block schema，未新增任何 MovL/MovJ |

## 2. 新增工程变量（`data.vars.project`，索引 15–20）

| 索引 | 名称 | 类型 | 初始值 | 用途 |
|------|------|------|--------|------|
| 15 | `p2_arrived` | STRING | `""` | CPosToStr 输出：P2 到位消息（前缀 `P2_ARRIVED\|`） |
| 16 | `p1_arrived` | STRING | `""` | CPosToStr 输出：P1 到位消息（前缀 `P1_ARRIVED\|`） |
| 17 | `grasp_signal` | STRING | `""` | SocketReadStr 接收：PC 对 P2 的授权回执 |
| 18 | `grip_signal` | STRING | `""` | SocketReadStr 接收：PC 对 P1 的授权回执 |
| 19 | `grasp_parse_flag` | INT | `0` | TranStrToCpos `<...>` 解析 P2 回执的结果 flag（0=成功） |
| 20 | `grip_parse_flag` | INT | `0` | TranStrToCpos `[...]` 解析 P1 回执的结果 flag（0=成功） |

## 3. 新增位姿点（`points.children`，索引 22–25）

| 索引 | 名称 | 引用者 | 用途 |
|------|------|--------|------|
| 22 | `H_P2_ACTUAL` | GetCurCPos（P2 屏障） | 运行时写入 P2 实际到达位姿；**不作为运动目标** |
| 23 | `H_P1_ACTUAL` | GetCurCPos（P1 屏障） | 运行时写入 P1 实际到达位姿；**不作为运动目标** |
| 24 | `DummyGraspAck` | TranStrToCpos `<...>`（P2 屏障） | 仅存 PC 授权回执；**未被任何 MovL/MovJ 引用** |
| 25 | `DummyGripAck` | TranStrToCpos `[...]`（P1 屏障） | 仅存 PC 授权回执；**未被任何 MovL/MovJ 引用** |

**确认**：已逐块扫描输出文件全部 `movl` 块（共 22 个，与来源一致），其 `pointa.pointvar.value` 集合不含 `ptmmsstjiwzzqzdmu8g`（DummyGraspAck）与 `ptmmsstjiwz281l1nbi`（DummyGripAck）。两个 Dummy 点不被任何运动块引用。

## 4. 新增 block 清单（widget 路径 + 用途）

抓壶段原序列 `MovL H_P2 → Wait → MovL H_P1 → SocketSendStr str2(GRIPDONE)` 现为：

```
[34] MovL H_P2（原块保留）
[35] Wait（原块保留）
[36]–[42] P2 屏障（新增）
[43] MovL H_P1（原块保留，仅 P2 屏障成功后可达）
[44]–[50] P1 屏障（新增）
[51] SocketSendStr str2=GRIPDONE（原块保留，仅 P1 屏障成功后可达）
```

### P2 屏障（widgets.children[36]–[42]）

| 路径 | 类型 | 作用 |
|------|------|------|
| `widgets.children[36]` | `getcurcpos` | 读取 P2 实际位姿 → `H_P2_ACTUAL` |
| `widgets.children[37]` | `cpostostr` | `H_P2_ACTUAL` → `p2_arrived`，前缀 `P2_ARRIVED\|`，后缀空，分隔符 `,`，m/deg |
| `widgets.children[38]` | `socketsendstr` | 发送 `p2_arrived`（Socket0）→ PC 收到 P2 到位回执请求 |
| `widgets.children[39]` | `socketreadstr` | 读 `grasp_signal`（Socket0），**15000ms 超时**，flag=`socketflag` |
| `widgets.children[40]` | `if` | `socketflag != 0` → **失败分支**：`SocketClose(Socket0)` + `Return`（不回 H_P1） |
| `widgets.children[41]` | `transtrtocpos` | `<...>` 解析 `grasp_signal` → `DummyGraspAck`，flag=`grasp_parse_flag` |
| `widgets.children[42]` | `if` | `grasp_parse_flag != 0` → **失败分支**：`SocketClose(Socket0)` + `Return` |

### P1 屏障（widgets.children[44]–[50]）

| 路径 | 类型 | 作用 |
|------|------|------|
| `widgets.children[44]` | `getcurcpos` | 读取 P1 实际位姿 → `H_P1_ACTUAL` |
| `widgets.children[45]` | `cpostostr` | `H_P1_ACTUAL` → `p1_arrived`，前缀 `P1_ARRIVED\|` |
| `widgets.children[46]` | `socketsendstr` | 发送 `p1_arrived`（Socket0） |
| `widgets.children[47]` | `socketreadstr` | 读 `grip_signal`（Socket0），**15000ms 超时**，flag=`socketflag` |
| `widgets.children[48]` | `if` | `socketflag != 0` → **失败分支**：`SocketClose(Socket0)` + `Return` |
| `widgets.children[49]` | `transtrtocpos` | `[...]` 解析 `grip_signal` → `DummyGripAck`，flag=`grip_parse_flag` |
| `widgets.children[50]` | `if` | `grip_parse_flag != 0` → **失败分支**：`SocketClose(Socket0)` + `Return` |

**成功路径**：`[43] MovL H_P1` 仅在 `[40][42]` 均未拒绝时到达；`[51] SocketSendStr(GRIPDONE)` 仅在 `[48][50]` 均未拒绝时到达。
**失败路径**（4 处，均 `SocketClose(Socket0)` + `Return`）：`[40]`（socket 超时/读失败）、`[42]`（回执非 `<...>` 格式）、`[48]`（socket 超时/读失败）、`[50]`（回执非 `[...]` 格式）。任一失败不会发送 `GRIPDONE`，不会继续任何运动。

## 5. 离线验证证据

```text
$ python3 -c "import json; json.load(open('doc/handlepluscup-orchestrator-session-barrier-2026-08-14.crp','r',encoding='utf-8')); print('JSON_OK')"
JSON_OK

$ sha256sum doc/handlepluscup-orchestrator-session-barrier-2026-08-14.crp
3d16aa59fa64fd6b6bf1985ed1306af5d827c9f8d3b0b09f4304646a4af810bf
```

**编辑器重新打开**：输出文件已在 Codroid/ESTUN 编辑器 v1.6.3b 中打开，工程正常加载，新增 block 与变量/位姿点正常显示，**未运行**。首次打开验证通过。

## 6. 硬件/服务状态声明

本次任务全程**未**执行以下任何操作：未启动 `Orchestrator/main.py` 或任何 Python/TCP 服务；未连接相机（RealSense/UGREEN）；未连接或使能灵巧手；未连接、导入、下发、单步或运行机械臂/控制器工程；未修改生产 CRP `handlepluscup-orchestrator.crp`（仅以只读方式作为克隆来源）。生成的 session-barrier 工程不含任何实际 I/O 启动入口，模板中全部 `MovL` 均为来源工程原有运动（22 个，未新增）。
