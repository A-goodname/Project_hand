# Python host session gate — 变更说明（2026-08-14）

对应指令：《project-hand-session-barrier-crp-v2-accepted-2026-08-14.md》"下一项离线实施指令：Python host session gate"
基准 commit：`2dd6fc9`（工作树含本变更，未提交）| 阶段：D0，仅离线实现与测试，未运行任何程序/硬件

## 1. 变更文件与 SHA-256

| 文件 | 状态 | SHA-256 |
|------|------|---------|
| `Orchestrator/session_barrier.py` | **新增** | `c78d5921804a5b15563b07779fc74487d8459aa4f8332d74a6522dc37c660af9` |
| `Orchestrator/state_machine.py` | 修改 | `751dd06d8daf3f0a9e6768c27b169ef4e5e6b54e69abbf94375701a24bb2a813` |
| `Orchestrator/handle_cup_detector.py` | 修改（+1 字段） | `aae72cb2d8f6abd46e49a1d2c37ffcf322bd0332f1e732e67caee69b5672c2c5` |
| `Orchestrator/tests/test_session_barrier.py` | **新增** | `4a6c03abfab1a48fc6e9dfa5c96d16343c337532e2009287ebc3abb564bde19a` |

## 2. 与指令 7 条的对照

### ① 标定失败关闭（`state_machine.py`）
- `load_cam2ee()` 重写：文件不存在 / YAML 无效 / 缺 `handeye` / 缺 `use_method` / 方法缺失 / 非 4×4 / 非有限值 / 非刚体（行列式 ∉[0.9,1.1] 或 R 非正交）/ 单位变换 / 样本数 <10 / 重投影误差 >0.5px → 抛 `CalibrationError`（新异常）
- **绝不返回 `np.eye(4)`**；不提供默认 Tsai 回退
- `_start()` 顺序调整：标定加载移到 TCP server 创建**之前**——标定失败则异常传播、`start()` 失败、**不监听运动 socket、不发候选位姿**
- 新增 `calib_fingerprint(T)`：标定摘要（平移量），会话内复核

### ② 保留 D435i 至 P2 复核结束
- 删除原"Phase B 后 `ctx.handle_cup_detector = None` / `T_cam2ee = None`"逻辑；仅 `targets_sent.set()` 通知 main 进入 seated_check，**pipeline 保留**
- 新增 `_refresh_handle_observation()`：P2 复核时**清空稳定器缓存 + 新代际**（旧观测作废），主动 `get_frames()` + `detect()`，等待新稳定把手观测（≤6s），返回观测时间戳
- 新增 `_release_camera_a_late()`：GRIPDONE 门通过（复核结束）或会话清理时，才把引用交给 main 线程关闭
- `HandleCupDetector` 新增 `last_known_pose_ts` 字段（+1 行初始化，`_detect` 更新处记录 `time.time()`）——"只接受时间戳严格晚于 P2 回执的观测"

### ③ PC 本地状态机（`session_barrier.py`，新文件）
- 每个 TCP 会话 = 一个新 `SessionBarrier` 实例（run_id = 时间戳+端口）
- 初始计划发送时 `issue_pregrasp()` 签发**仅内存**的 pregrasp ticket
- `WAIT_P2_ARRIVED`：仅接受 `P2_ARRIVED|x,y,z,a,b,c` → 严格解析（ASCII/无空白/6 有限数）→ 位置欧氏距离 ≤0.02m、每轴角度差 ≤5deg（可配）→ pregrasp ticket 未过期（TTL 30s）→ 标定摘要一致 → **fresh_obs_ts > P2 回执时刻**（新鲜 D435i 观测）→ 全部通过才签发 grasp ticket 并返回 `<P0>`
- `WAIT_P1_ARRIVED`：仅接受 `P1_ARRIVED|...` → 复核 → **同一 grasp ticket 未消费才回 `[P0]` 并消费**
- `WAIT_GRIPDONE`：仅精确 `GRIPDONE` → 返回 GRIPDONE 信号（state_machine 此刻才调 `hand_grip_done()` 一次，`_grip_done_once` 防重）

### ④ 一次性消费
- grasp ticket `consumed` 标记 + 重放检测（重复 P1 回执 → `GRASP_REPLAY`）
- `GRIPDONE` 重复/越序 → 拒绝；屏障通过后旧流程的 GRIPDONE 分支也由 `_grip_done_once` 拦截

### ⑤ 失败关闭
- 空/非 ASCII/含空白/非 6 有限数/前缀错 → `BarrierReject`（含原因码：`EMPTY/NON_ASCII/WHITESPACE/BAD_PREFIX/BAD_COUNT/BAD_NUMBER/NOT_FINITE/SEQ/P2_TOL/P1_TOL/STALE_OBS/PG_*/GRASP_REPLAY/...`）
- 任何拒绝 → `_cleanup()`：撤销 ticket（grasp 置 consumed）、关闭连接、`stop_event.set()`、释放 Camera A——**不发 ACK、不进 H_P1、不闭合**
- `PREGRIP` 仅手势（`PREGRIP` action 返回，状态不变，不绕过屏障）

### ⑥ P0 ACK 限制
- `<P0>` / `[P0]` = 本会话 Phase A 已验证的实际位姿 `(robot_xyz, robot_ABC)`，米/度，3/2 位小数；**Phase A 位姿全零 → 会话直接拒绝**（不创建 barrier）

### ⑦ 范围冻结
- 未改迎宾（GreetingHandler）、递杯/HandInCup、H_P1-H_P4 点位、速度、倒水、相机端口、灵巧手端口
- 生产 CRP `handlepluscup-orchestrator.crp` 与 v2 文件均未改动；`state_machine.py` 仅抓壶 Phase B/C 接入

## 3. 离线验证证据

```text
$ python3 -m py_compile Orchestrator/session_barrier.py Orchestrator/state_machine.py \
    Orchestrator/handle_cup_detector.py Orchestrator/tests/test_session_barrier.py
(无输出 = 通过)

$ python3 -m unittest Orchestrator.tests.test_session_barrier -v
Ran 22 tests in 0.011s — OK
(覆盖: 解析 8 例 / 容差 3 例 / 状态机 11 例 — 全流程通过、越序、格式错、容差、ticket 过期、旧 ticket、
 新鲜观测缺失、重放、PREGRIP 手势、P0 非零回执)
```

## 4. 硬件/服务状态声明

本次**未**启动 `Orchestrator/main.py`、未启动任何 TCP 服务、未连接相机/灵巧手/机械臂、未导入或运行任何 CRP。全部验证为纯 Python 语法检查与内存内单元测试（`SessionBarrier` 纯逻辑，无 socket/硬件依赖）。
