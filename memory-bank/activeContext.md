# Active Context — Project_hand

## 当前焦点

**单相机状态机重构 + CRP 生成** — 从 Orchestrator/state_machine.py 移植检测+位姿计算, 仅保留 estun_* 的运动流程.

## 最近变更

### VLA 数据采集版状态机 (2026-07-31)

1. **`Orchestrator/estun_handle_data_sm.py`** — 新建, 数据采集版 (VLA 训练). 检测器含 Cup 检测 → 用杯位算 Handle P3 倒水点; Phase A→B→C; Phase C 仅 PREGRIP/GRIPDONE/RESET (无 CUP/RELEASE/GRIP).
2. **`doc/estun_handle_data_sm.crp`** — 35 widgets, 8 points. 从 estun_handle_cup_sm.crp 克隆: 删 Cup 段, 移植旧 orchestrator CRP 的 **H_P4 回退避碰点** (CposOffset: H_P4 = H_P1 + DCp1, DCp1 = H_P3−H_P1 − Retreat, Retreat 按倒水角 B 动态: <25°→150mm, <17.5°→175mm, <12.5°→200mm).
   - 运动序列: Home → P2 → P1(抓) → H_P4(V100) → H_P3(倒水,sudu) → H_P4(sudu) → P1 → RESET → P2 → Home
   - Python 只发 3 位姿 (pack_three), H_P4 由 CRP 内 CposOffset 动态计算
3. **窗口标题修复** — `estun_handle_sm.py`/`estun_handle_cup_sm.py` 中 em dash `—` → ASCII `|` (Windows GBK 下 namedWindow/imshow 编码不一致导致双窗口+乱码).
4. **生成脚本** — `scripts_gen/gen_handle_data_crp.py` (skill 规范: 备份 .bak + estun_crp_helpers.new_id/write_json; 支持 `--keep-cup` 参数).
5. **`doc/estun_handle_cup_sm.crp` 补 H_P4** — 53 widgets/11 points. Handle 倒水段加入 H_P4 避碰 (与 data 版相同 DCp1 计算), Cup 段完整保留.
6. **`doc/estun_handle_sm.crp` P3 速度** — H_P3 MovL `sudu` → `V100` (P3 仅 Z+5cm 抬升, 无需倒水低速).

CRP 现状 (2026-07-31):
| 文件 | widgets | H_P4 | 说明 |
|------|---------|------|------|
| estun_handle_sm.crp | 23 | 无 | 仅水壶, P3=Z-up, 全 V100 |
| estun_handle_cup_sm.crp | 53 | 有 | Handle 倒水避碰 + Cup 段 |
| estun_handle_data_sm.crp | 35 | 有 | 数据采集, 无 Cup 段 |

### 单相机状态机 + CRP (2026-07-31)

1. **`Orchestrator/estun_handle_sm.py`** — 新建, 仅水壶抓取. 检测器=HandleCupDetector(轻量), 位姿=固定ABC, P3=Z-up.
2. **`Orchestrator/estun_handle_cup_sm.py`** — 新建, 水壶+水杯. 同上架构, Cup P3=Z-up(cup_lift_z).
3. **`doc/estun_handle_sm.crp`** — 23 widgets, 对应仅水壶流程.
4. **`doc/estun_handle_cup_sm.crp`** — 41 widgets, 对应水壶+水杯流程.
5. **`Orchestrator/config.yaml`** — 新增 `handle_lift_z: 0.05` + `cup_lift_z: 0.05`.
6. **`Orchestrator/handle_cup_detector.py`** — `detect()` 改为红色优先 (先在 kettle 右侧搜红色胶带, 未找到才回退白色).
7. **白色把手检测两遍法**: Pass1 严格(原阈值) → Pass2 宽松(暗光), 已合并到 `find_white_handle_roi`.

### 架构差异

| 维度 | 旧 estun_handle_cup_run.py | 新 estun_handle_cup_sm.py |
|------|--------------------------|--------------------------|
| 检测器 | RealSenseHandle6D + RealSenseCup6D (SAM/PointNet++) | HandleCupDetector (YOLO+HSV 轻量) |
| 位姿 | 动态计算 (compute_grasp_rotation + post_correct) | 全部固定 ABC (Orchestrator/config.yaml) |
| 状态管理 | 模块级全局变量 ×15 | 类封装 (HandleCupSM) |
| Phase | A→B→C (单体 _handle_client_impl) | A→B→C (3 个独立方法) |
| Cup P3 | 复杂 target_offset (EE 系旋转+平移) | P1 + Z-up (基系直加) |

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | CRP 实机导入测试 | `estun_handle_sm.crp` + `estun_handle_cup_sm.crp` 导入示教器 |
| 🔴 | 单相机 SM 实机联调 | `estun_handle_sm.py` + `estun_handle_cup_sm.py` 连机械臂 |
| 🟡 | LLM 语音控制实机联调 | 连 OmniHand |
| 🟡 | 加水实测 | 完整流程 |
