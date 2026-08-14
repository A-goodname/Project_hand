# handlepluscup-orchestrator-session-barrier-v2 离线模板 manifest（2026-08-14）

对应返工单：`project-hand-session-barrier-crp-audit-reject-2026-08-14.md`（v1 六处字段修正：4×`lengthunit` mm→m、2 组 send/read `Socket1`→`Socket0`）
阶段：D0，仅离线编辑与回传。本次未连接/使能/运行任何硬件。

## 1. 文件与版本

| 项 | 值 |
|----|-----|
| 来源文件 | `doc/handlepluscup-orchestrator.crp`（锁定源） |
| 来源 SHA-256 | `342cb0e971773d9b113a5bdd0a108b36b1bb17fa9cdf225b8fa5adc218f5bd3a` |
| v1 文件（被拒） | `handlepluscup-orchestrator-session-barrier-2026-08-14.crp`，SHA `3d16aa59...`（未覆盖，保留） |
| **v2 输出文件** | `doc/handlepluscup-orchestrator-session-barrier-v2-2026-08-14.crp` |
| **v2 SHA-256** | `1e109d57a8f0a5f16d3bce92513767c324c6374cd5240efc0ebe8b24c215d31f` |
| v2 大小 | 207258 字节 |
| 编辑器版本 | **v1.6.3b**（Codroid/ESTUN 编辑器） |
| 控制器/固件版本 | `unknown`（本机无示教器可读） |

## 2. 相对 v1 的修正（返工单要求，已全部落实）

| widget 路径 | 字段 | v1 实际值 | v2 值 |
|------|------|-----------|-------|
| `widgets.children[37]` P2 `CPosToStr` | `lengthunit` | `mm` | **`m`** |
| `widgets.children[41]` P2 ACK `TranStrToCpos` | `lengthunit` | `mm` | **`m`** |
| `widgets.children[45]` P1 `CPosToStr` | `lengthunit` | `mm` | **`m`** |
| `widgets.children[49]` P1 ACK `TranStrToCpos` | `lengthunit` | `mm` | **`m`** |
| `widgets.children[38]`, `[39]` P2 `SocketSendStr`/`SocketReadStr` | `mscname` | `Socket1` | **`Socket0`** |
| `widgets.children[46]`, `[47]` P1 `SocketSendStr`/`SocketReadStr` | `mscname` | `Socket1` | **`Socket0`** |

`angleunit="deg"`、分隔符 `,`、`P2_ARRIVED|`/`P1_ARRIVED|` 前缀、15000ms timeout、`socketflag != 0`/`grasp_parse_flag != 0`/`grip_parse_flag != 0` 的 `SocketClose(Socket0) + Return` 分支均与 v1 保持一致。未调整任何点位、速度、手部动作、H_P3/H_P4、迎宾、递杯或既有业务逻辑。

## 3. 新增工程变量（`data.vars.project`，索引 15–20）

| 索引 | 名称 | 类型 | 初始值 | 用途 |
|------|------|------|--------|------|
| 15 | `p2_arrived` | STRING | `""` | CPosToStr 输出：P2 到位消息（前缀 `P2_ARRIVED\|`） |
| 16 | `p1_arrived` | STRING | `""` | CPosToStr 输出：P1 到位消息（前缀 `P1_ARRIVED\|`） |
| 17 | `grasp_signal` | STRING | `""` | SocketReadStr 接收：PC 对 P2 的授权回执 |
| 18 | `grip_signal` | STRING | `""` | SocketReadStr 接收：PC 对 P1 的授权回执 |
| 19 | `grasp_parse_flag` | INT | `0` | TranStrToCpos `<...>` 解析 P2 回执结果 flag（0=成功） |
| 20 | `grip_parse_flag` | INT | `0` | TranStrToCpos `[...]` 解析 P1 回执结果 flag（0=成功） |

## 4. 新增位姿点（`points.children`，索引 22–25）

| 索引 | 名称 | id | 引用者 | 用途 |
|------|------|-----|--------|------|
| 22 | `H_P2_ACTUAL` | `ptmmsstxgjddd9z9h0w` | GetCurCPos（P2 屏障） | 运行时写入 P2 实际位姿；不作为运动目标 |
| 23 | `H_P1_ACTUAL` | `ptmmsstxgjdzktbptu8` | GetCurCPos（P1 屏障） | 运行时写入 P1 实际位姿；不作为运动目标 |
| 24 | `DummyGraspAck` | `ptmmsstxgjd42lsq14g` | TranStrToCpos `<...>`（P2 屏障） | 仅存 PC 授权回执；**未被任何 MovL/MovJ 引用** |
| 25 | `DummyGripAck` | `ptmmsstxgjdidjl0djl` | TranStrToCpos `[...]`（P1 屏障） | 仅存 PC 授权回执；**未被任何 MovL/MovJ 引用** |

**确认**：已逐块扫描 v2 全部 `movl`（22 个，与来源一致），`pointa.pointvar.value` 集合不含 `ptmmsstxgjd42lsq14g` 与 `ptmmsstxgjdidjl0djl`。两个 Dummy 点不被任何运动块引用。

## 5. 新增 block 清单（widget 路径 + 关键参数 + 用途）

抓壶段序列：`[34] MovL H_P2（原块）→ [35] Wait（原块）→ [36]–[42] P2 屏障（新增）→ [43] MovL H_P1（原块）→ [44]–[50] P1 屏障（新增）→ [51] SocketSendStr str2=GRIPDONE（原块）`

### P2 屏障（widgets.children[36]–[42]）

| 路径 | 类型 | 关键参数 | 作用 |
|------|------|----------|------|
| `widgets.children[36]` | `getcurcpos` | → `H_P2_ACTUAL` | 读取 P2 实际位姿 |
| `widgets.children[37]` | `cpostostr` | 输入 `H_P2_ACTUAL` → `p2_arrived`；前缀 `P2_ARRIVED\|`，后缀空；`separate=,`；`lengthunit=m`；`angleunit=deg` | 构造 P2 到位消息 |
| `widgets.children[38]` | `socketsendstr` | `mscname=Socket0`；发送 `p2_arrived` | 上报 P2 到位 |
| `widgets.children[39]` | `socketreadstr` | `mscname=Socket0` → `grasp_signal`；`mtesttime=15000`；flag=`socketflag` | 等 PC 授权回执（15s 有限超时） |
| `widgets.children[40]` | `if` | `socketflag != 0` → `SocketClose(Socket0)` + `Return` | 失败分支：读失败/超时拒绝 |
| `widgets.children[41]` | `transtrtocpos` | 输入 `grasp_signal` → `DummyGraspAck`；`strstring=<` `endstring=>`；`separate=,`；`lengthunit=m`；`angleunit=deg`；flag=`grasp_parse_flag` | 校验回执为 `<...>` 格式 |
| `widgets.children[42]` | `if` | `grasp_parse_flag != 0` → `SocketClose(Socket0)` + `Return` | 失败分支：回执格式错误拒绝 |

### P1 屏障（widgets.children[44]–[50]）

| 路径 | 类型 | 关键参数 | 作用 |
|------|------|----------|------|
| `widgets.children[44]` | `getcurcpos` | → `H_P1_ACTUAL` | 读取 P1 实际位姿 |
| `widgets.children[45]` | `cpostostr` | 输入 `H_P1_ACTUAL` → `p1_arrived`；前缀 `P1_ARRIVED\|`；`lengthunit=m`；`angleunit=deg` | 构造 P1 到位消息 |
| `widgets.children[46]` | `socketsendstr` | `mscname=Socket0`；发送 `p1_arrived` | 上报 P1 到位 |
| `widgets.children[47]` | `socketreadstr` | `mscname=Socket0` → `grip_signal`；`mtesttime=15000`；flag=`socketflag` | 等 PC 授权回执（15s 有限超时） |
| `widgets.children[48]` | `if` | `socketflag != 0` → `SocketClose(Socket0)` + `Return` | 失败分支：读失败/超时拒绝 |
| `widgets.children[49]` | `transtrtocpos` | 输入 `grip_signal` → `DummyGripAck`；`strstring=[` `endstring=]`；`lengthunit=m`；flag=`grip_parse_flag` | 校验回执为 `[...]` 格式 |
| `widgets.children[50]` | `if` | `grip_parse_flag != 0` → `SocketClose(Socket0)` + `Return` | 失败分支：回执格式错误拒绝 |

**成功路径**：`[43] MovL H_P1` 仅在 `[40][42]` 均未拒绝时到达；`[51] SocketSendStr(GRIPDONE)` 仅在 `[48][50]` 均未拒绝时到达。
**失败路径**（4 处，均 `SocketClose(Socket0)` + `Return`）：`[40]`、`[42]`、`[48]`、`[50]`。任一失败不发送 GRIPDONE、不继续运动。

## 6. 离线验证证据

```text
$ python3 -c "import json; json.load(open('doc/handlepluscup-orchestrator-session-barrier-v2-2026-08-14.crp','r',encoding='utf-8')); print('JSON_OK')"
JSON_OK

$ sha256sum doc/handlepluscup-orchestrator-session-barrier-v2-2026-08-14.crp
1e109d57a8f0a5f16d3bce92513767c324c6374cd5240efc0ebe8b24c215d31f
```

**编辑器重新打开**：v2 文件已在 Codroid/ESTUN 编辑器 v1.6.3b 中打开，工程正常加载，六组修正字段与新增 block/变量/位姿点正常显示，保存后关闭并再次打开成功，**未运行**。

## 7. 硬件/服务状态声明

本次返工全程**未**执行：未启动 `Orchestrator/main.py` 或任何 Python/TCP 服务；未连接相机（RealSense/UGREEN）；未连接或使能灵巧手；未连接、导入、下发、单步或运行机械臂/控制器工程；未修改生产 CRP `handlepluscup-orchestrator.crp` 与被拒 v1 文件（均仅只读）。v2 不含任何实际 I/O 启动入口，全部 `MovL`（22 个）为来源工程原有，未新增。
