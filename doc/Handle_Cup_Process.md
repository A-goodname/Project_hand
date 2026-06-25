# 水杯抓取与把手抓取融合 (Handle + Cup)

## 上次更新: 2026-06-25 (v5: 通信协议重构 + P4 预倒水点)

---

## 1. 概述

将水壶把手抓取与水杯抓取融合，实现全流程自动化：

1. **Phase 1 — 抓水壶 + 倒水**: 检测水壶把手 → 抓取 → 移动到倒水位置 → 倒水
2. **Phase 2 — 抓水杯 + 递杯**: 检测水杯 → 抓取 → 移动到递杯位置 → 释放

共 6 个机器人位姿 (Handle P1/P2/P3_pour + Cup P1/P2/P3), 分两阶段发送.

**v4 核心变更**: 倒水口不再由 cup 检测器直接输出 (移除 Phase 3/4 的 PCA rim 检测), 而是基于 Cup TCP 在基系下的位置 `c_pos_base_1` + 杯几何偏移 (rim_radius, height) 计算得到. 修复了透明杯子深度误差导致倒水口位置不准的问题.

---

## 2. 坐标系与符号约定

| 符号 | 含义 |
|------|------|
| P_obj_cam | 物体在相机坐标系下的 6D 位姿 (检测器输出) |
| T_cam_ee | 相机到末端法兰的 4x4 齐次变换矩阵 (手眼标定) |
| P_ee_base | 当前末端法兰在基坐标系下的位姿 [x,y,z,A,B,C] |
| T_ee_base | 从末端法兰到基坐标系的 4x4 齐次矩阵 |
| t_tool_ee / r_tool_ee | 工具相对末端的平移/旋转偏移 |
| c_pos_base_1 | Cup TCP 在基系下的位置 (含 cup gripper_offset 补偿) |
| pour_target_base | 倒水口在基系下的位置, 由 `c_pos_base_1 + [rim_radius, 0, height/2]` (back 模式) 计算 |
| CUP_RIM_RADIUS | 杯口半径先验, 从 `config.yaml cup.rim_radius` 读取 (默认 0.035 m) |
| CUP_HEIGHT | 杯高先验, 从 `config.yaml cup.height` 读取 (默认 0.100 m) |

---

## 3. Phase 1 — 把手抓取 + 倒水

### 3.1 流程

```
MoveP2(handle) → PREGRIP → MoveP1(handle) → GRIPDONE → MoveP3(handle_pour) → wait 1s(倒水)
→ MoveP1(handle) → PREGRIP → MoveP2(handle) → RESET → MoveHome
```

### 3.2 Handle P1 (抓取点)

与 `estun_handle_run` 完全一致:
1. 检测水壶 Handle 6D → cam2ee → 稳定化 → 得到 `pos_ee, euler_ee`
2. 位置补偿: `pos_ee_compensated = pos_ee - t_handle_effective`
3. 方向对齐: `R_grasp_ee = compute_grasp_rotation(R_obj_ee, mode, r_handle_offset, R_ee_base)`
4. 转到基系: `pos_base_1 = transform_pose_to_base(pos_ee_compensated, robot_xyz, robot_ABC)`
5. `A1,B1,C1` 由 `rot_base_1 = R_ee_base @ R_grasp_ee` 解算 + 可达性修正

### 3.3 Handle P2 (预抓取点)

与 `estun_handle_run` 完全一致:
- 位置: `pos_ee_pre = pos_ee_compensated + R_grasp_ee_corrected @ HANDLE_PRE_T_OFFSET`
- 姿态: 继承 P1 修正后的 `(A1, B1, C1)`

### 3.4 Handle P3_pour (倒水位置) — v4 重构

P3 计算不再依赖 cup 检测器输出的 `pour_target` (PCA rim 中心), 而是基于 **Cup TCP 基系位置** `c_pos_base_1`, 该位置已包含 cup gripper_offset 补偿, 对透明杯子更准确.

#### 计算顺序

由于 P3 需要 Cup TCP 位置, 计算顺序调整为:

```
Phase 1a: Handle P1/P2 (不依赖倒水口)
Phase 2:  Cup P1/P2/P3 (得到 c_pos_base_1 = Cup TCP 基系位置)
Phase 1b: 基于 c_pos_base_1 计算倒水口 → Handle P3
```

#### 倒水口 (pour_target) 计算

倒水口在基系下 = Cup TCP 位置 + 杯几何偏移:

```python
if CUP_HANDLE_SIDE == "back":
    pour_target_base = c_pos_base_1 + np.array([CUP_RIM_RADIUS, 0.0, CUP_HEIGHT / 2.0])
else:  # left
    pour_target_base = c_pos_base_1 + np.array([0.0, CUP_RIM_RADIUS, CUP_HEIGHT / 2.0])
```

- `back` 模式: 杯口中心在 Cup TCP 的 X 正方向 rim_radius 处, Z 方向向上 height/2
- `left` 模式: 杯口中心在 Cup TCP 的 Y 正方向 rim_radius 处, Z 方向向上 height/2

然后将 `pour_target_base` 反算回相机系 (供 Handle P3 cam2ee 链路使用):

```python
pour_target_ee_calc = R_ee_base.T @ (pour_target_base - t_ee_base)
pour_target = T_cam2ee[:3, :3].T @ (pour_target_ee_calc - T_cam2ee[:3, 3])
```

#### Handle P3 计算

**1. 倒水位置 → 末端系:**
```
pos_pour_ee = T_cam2ee[:3,:3] @ pour_target + T_cam2ee[:3,3]
```

**2. 反算抓取点末端位姿 (fixed 模式):**
```python
pos_ee_p3 = pos_pour_ee - R_grasp_ee @ SPOUT_T_OFFSET - t_handle_effective
pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) + POUR_T_OFFSET_BASE
rot_base_3 = R_ee_base @ R_target_ee
```

**3. 反算抓取点末端位姿 (vertical/free 模式):**
```python
spout_rotated = R_pour_ee @ SPOUT_T_OFFSET  # R_pour_ee = R_grasp_ee @ POUR_R_OFFSET
pos_ee_p3 = pos_pour_ee - spout_rotated - t_handle_effective
pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) + POUR_T_OFFSET_BASE
rot_base_3 = R_ee_base @ R_pour_ee
```

**关键修复 (对比旧代码):**
- P3 位置加入了 `t_handle_effective` 补偿 (旧代码遗漏)
- vertical/free 模式使用 `_R_pour_ee @ SPOUT_T_OFFSET` 而非 `_R_grasp_ee_corrected @ SPOUT_T_OFFSET` (包含倒水旋转)

---

## 4. Phase 2 — 水杯抓取 + 递杯

### 4.1 流程

```
(发送 'CUP' 标记)
MoveP2(cup) → RELEASE(PAPER) → MoveP1(cup) → GRIP → MoveP3(cup) → wait 1s(等人取)
→ MoveP1(cup) → RELEASE(PAPER) → MoveP2(cup) → RESET → MoveHome
```

### 4.2 Cup P1 (抓取点)

1. `RealSenseCup6DDetector.compute_cup_6d_pose()` 输出 (新版 5 元组):
   ```
   (bbox, position, rotation, euler, point_cloud)
   ```
   - 已移除 `rim_pos`, `pour_target`, `rim_r` 输出 (倒水口改用几何计算)
2. `position` 是水杯在相机系下的 3D 位置 (YOLO bbox 重心深度 + PCA 均值)
3. cam2ee 转换:
   ```
   pos_cup_ee = T_cam2ee[:3,:3] @ position + T_cam2ee[:3,3]
   rot_cup_ee = T_cam2ee[:3,:3] @ rotation
   ```
4. 位置补偿 (使用 `cup_grasp.gripper_offset`, 根据 `handle_side` 选择偏移):
   ```python
   if CUP_HANDLE_SIDE == "left":
       t_cup_effective = R_z(r_cup_offset.C) @ t_cup_offset_left
   else:
       t_cup_effective = R_z(r_cup_offset.C) @ t_cup_offset
   pos_ee_compensated = pos_cup_ee - t_cup_effective
   ```
5. 方向对齐 + 基系转换
6. **角度约束** (根据 handle_side):
   - `back` 模式: B 角 clamp 到 [70, 90], C 角 clamp 到 [-120, -100]
   - `left` 模式: C 角 clamp 到 [-100, -80], 默认 -90

### 4.3 Cup P2 (预抓取点)

```
pos_ee_pre = pos_ee_compensated + R_grasp_ee_cup @ CUP_PRE_T_OFFSET
姿态同 P1
```

### 4.4 Cup P3 (递杯点)

```
pos_ee_target = pos_ee_compensated + R_grasp_ee_cup @ CUP_TARGET_T_OFFSET
R_target_ee = R_grasp_ee_cup @ euler_to_rotation_matrix(*CUP_TARGET_R_OFFSET, order="zyx")
```

---

## 5. TCP 通信协议

基于 `estun_handle_run` 的 Phase C 指令循环扩展.

### 协议格式 (与 estun_handle_run 兼容)

```
[Phase B] 发送 handle P1/P2/P3:  (...) [...] {...}
[Phase C] 指令循环:
    PREGRIP → 灵巧手 PREGRIP
    GRIPDONE → 灵巧手 GRIPDONE
    RESET    → 灵巧手 RESET
    CUP      → 回复 cup P1/P2/P3  (格式同上)
                然后继续指令循环 (PREGRIP/GRIPDONE/RESET)
```

CUP 命令由机械臂在 handle 流程完成后发送. 回复 cup 位姿时, 直接 `sendall` 三个位姿, 格式与 Phase B 一致.

### 机械臂端流程 (v2, 2026-06-25)

详见 `doc/estun_handle_cup_code.txt`. 重构为线性五阶段流程:

```
[1] SocketSendStr(当前位姿) 一次, 在 while 循环外
[2] while SocketReadStr → 等待 Handle P1/P2/P3 (独立循环, 超时重试, 只读不写)
[3] Handle 运动: P2 → P1 → P4 → P3 → P4 → P1 → P2
[4] SocketSendStr('CUP') → while SocketReadStr → 等待 Cup P1/P2/P3
[5] Cup 运动: C_P2 → C_P1 → C_P3 → C_P1 → C_P2 → Home
```

**关键改进**:

- `SocketSendStr` 不再在 while 循环内反复发送 (避免 TCP buffer 污染)
- 两个 `SocketReadStr` 各自独立循环, 超时只重试不发送多余数据
- P4 预倒水点由机械臂端 `CposOffset` 计算, PC 端无需额外发送

### PC 端位姿组装 (v2)

抽取 `fmt_pose()` / `pack_three()` 函数, 统一为 `(P1)[P2]{P3}` 格式. 修正了旧代码 Cup P1 括号使用 `[...]` 而非 `(...)` 的 bug.

---

## 6. 检测器架构 (共享 pipeline)

### 6.1 冲突问题

`BaseRealSenseDetector.__init__` 默认启动 `rs.pipeline`. 若 Handle 和 Cup 两个检测器各自独立初始化, 第二个检测器启动时第一个的 pipeline 仍在运行, 导致设备冲突.

### 6.2 解决方案

1. **基类改造**: 添加 `start_pipeline=True` 参数
2. **Cup 检测器**: demo/run 中传入 `start_pipeline=False`, 不启动 pipeline
3. **帧注入**: 通过 `inject_frames()` 将 Handle 检测器获取的帧注入 Cup 检测器
4. **独立运行**: `python test_cup_6D.py` 时仍默认 `start_pipeline=True`

### 6.3 主循环流程

```
每 SKIP_FRAMES 帧:
  1. 从 Handle 检测器获取帧 (color, depth, depth_colormap, depth_frame)
  2. Handle 检测: detect_handle_6d(color, depth_frame)
  3. 注入帧到 Cup 检测器: cup_detector.inject_frames(...)
  4. Cup 检测: cup_detector.compute_cup_6d_pose(color, depth_frame)
  5. 分别稳定 Handle 和 Cup 位姿
```

---

## 7. 配置参数说明

### 7.1 `Estun/estun_config.yaml` — cup_grasp 段

| 参数 | 值 | 说明 |
|------|-----|------|
| `enabled` | true | 总开关 |
| `spout_offset.t_offset` | [-0.06, 0.17, 0.0] | 壶嘴在抓取末端系下的偏移 (m), 已实际测量 |
| `gripper_offset.t_cup_offset` | [0.06, -0.04, 0.08] | 抓水杯的灵巧手平移偏移 (m), back 模式 |
| `gripper_offset.t_cup_offset_left` | [0.06, 0.04, 0.08] | 抓水杯的灵巧手平移偏移 (m), left 模式 |
| `gripper_offset.r_cup_offset` | [0, 0, -90] | 灵巧手安装角度 (度) |
| `pre_offset.t_offset` | [-0.0, 0.0, -0.05] | 预抓取平移偏移 (m) |
| `target_offset.t_offset` | [-0.2, 0, 0] | 递杯平移偏移 (m) |
| `target_offset.r_offset` | [0, 0, 0] | 递杯旋转偏移 (度) |
| `pour_offset.t_offset` | [0, 0, 0.05] | 基系 Z 轴防碰撞偏移 (m) |
| `pour_offset.r_offset` | [0, 0, -45] | 倒水旋转偏移 (度, 同 handle target) |

### 7.2 `RealSense/config.yaml` — cup 段 (注意: v4 后统一使用 RealSense/ 新模块)

| 参数 | 值 | 说明 |
|------|-----|------|
| `height` | 0.100 | 杯高先验 (m), 实际测量 |
| `rim_radius` | 0.035 | 杯口半径先验 (m), 实际测量 |
| `rim_offset_ratio` | 0.6 | 倒水偏移比 (0~1), v4 后不再用于倒水口计算, 保留兼容 |
| `handle_side` | "back" | 杯把手方向: back(默认) / left |

### 7.3 杯口方向确定

注: v4 已将 rim/pour 检测从 cup 检测器中移除. 杯口方向不再由 Phase 3/4 PCA 确定, 而是通过 `cup_handle_side` 配置和杯几何参数在基系下直接计算 (见 §3.4).

---

## 8. 关键设计决策

### 8.1 倒水位置计算 (v4 重构)

**旧方案 (v3):** cup 检测器通过 PCA rim 检测输出 `pour_target` (相机系点云杯口中心), 直接用于 Handle P3. 透明杯子深度测量误差导致倒水口偏移, 该误差被 `t_cup_effective` 的 Z 分量掩盖.

**新方案 (v4):** 倒水口基于 Cup TCP 基系位置 `c_pos_base_1` + 杯几何偏移:
```
pour_target_base = c_pos_base_1 + [rim_radius, 0, height/2]  (back 模式)
```
- `c_pos_base_1` 已包含 cup gripper_offset 补偿, 位置准确
- 偏移量由用户实测的先验参数 `rim_radius`, `height` 决定, 不受深度测量误差影响

### 8.2 Cup 6D 检测器简化

v4 移除 `compute_cup_6d_pose()` 的 rim/pour 相关输出, 返回 5 元组:
```python
(bbox, position, rotation, euler, point_cloud)
```
- 移除 `estimate_rim_position()` 调用 (Phase 3)
- 移除 pour_target 计算 (Phase 4)
- `visualize_cup()` 签名同步简化

### 8.3 Phase C 指令扩展

不改变原有指令循环结构, 仅添加 `CUP` 命令处理. 机械臂端在 handle 流程结束后发送 `CUP`, 本机回复 cup 三个位姿.

### 8.4 同步检测

Handle 和 Cup 检测使用同一帧数据 (在 Handle 检测帧时同步执行 Cup 检测), 避免两阶段检测的帧不一致问题.

### 8.5 灵巧手手势映射 (函数式重构)

2026-06-11 将 `HandController` 类重构为 `Estun/utils_shared.py` 的函数式接口, 统一三套抓取流程:

| 函数 | 手势 | 适用场景 |
|------|------|----------|
| `init_hand(initial_gesture="PAPER")` | PAPER/RESET | 初始化, 水瓶默认 PAPER, 把手/融合默认 RESET |
| `hand_pre_grip()` | PREGRIP | 把手半闭合接近阶段 |
| `hand_grip_done()` | GRIPDONE | 把手完全夹紧 |
| `hand_reset()` | RESET | 复位灵巧手 (不关闭串口) |
| `grip_object()` | GRIP | 水瓶/水杯抓取 |
| `release_object()` | PAPER | 水瓶/水杯释放/预抓取张开 |
| `disable_hand()` | RESET→disable→close | 关闭灵巧手 |

### 8.6 当前状态

- `test_cup_6D.py`: 已验证通过, 使用新版 5 元组输出, rim/pour 检测已移除
- `Estun/estun_handle_cup_demo.py`: 已完成 v4 重构 + B/C 角防跳变修正, back 模式已验证通过
- `Estun/estun_handle_cup_run.py`: v4 重构 + B/C 角防跳变修正 + 通信协议重构 (v2, 2026-06-25): `fmt_pose()`/`pack_three()` 抽函数, 清零时机改为断开时, 自动重稳定流程
- `doc/estun_handle_cup_code.txt`: v2 线性五阶段流程 + P4 预倒水点 + 独立 while 循环

---

## 9. B/C 角约束与方向模式

### 9.1 模式定义

| 模式 | B/C 角修正 | 适用场景 |
|------|-----------|----------|
| `vertical` | B 角防跳变 + C 角区间 clamp | 侧面抓取竖直水瓶/水壶 (默认) |
| `free` | 仅 post_correct_angles 可达性修复 | PCA 稳定, 姿态由检测决定 |
| `fixed` | 使用配置 `fixed_abc` 固定 ABC | PCA 失效时的兜底姿态 |

### 9.2 B 角防跳变策略 (仅 vertical)

灵巧手右手抓取侧, B 角在 90° 附近跳变时导致灵巧手翻转. `estun_handle_cup_demo/run` + `estun_handle_demo/run` 统一实现:

```python
if ORIENTATION_MODE == "vertical" and GRASP_DIRECTION == "right":
    # C 角 clamp: Handle [-100, -80]; Cup back [-120, -100]; Cup left [-100, -80]
    # B 角防跳变:
    if B < 90:  B = 89.0
    if B > 90:  B = 91.0
    # B == 90 时不修正 (机械臂特殊位姿)
```

**free 模式** 不做任何 B/C 角修正, 仅通过 `post_correct_angles()` 做 A/C 角 ±180° 可达性修复.

**fixed 模式** 使用配置固定 ABC, 不存在跳变问题.

### 9.3 P4 预倒水点

**动机**: Handle P1→P3 运动时若水杯与水壶接近, 高位直接 Z 向运动可能碰撞.

**定义**: `P4 = [P1.x, P1.y, P3.z, P1.a, P1.b, P1.c]` — 继承 P1 的 XY 和姿态, 取 P3 的 Z 高度.

**路径**: `P2 → P1 → P4 → P3 → P4 → P1 → P2`

**实现**: 由机械臂端 CposOffset 实现, Python 侧不额外发送位姿. 适用于 P3_TEACH_ENABLE=true (示教模式) 和视觉模式.

**当前状态**: 已完成. 机械臂端代码见 `doc/estun_handle_cup_code.txt` (v2), 待在实际示教器上测试.

---

## 10. 待办事项

- [x] P4 预倒水点: 机械臂端代码已实现 (`doc/estun_handle_cup_code.txt` v2), 待示教器测试
- [x] PC 端通信协议重构: `fmt_pose()`/`pack_three()` + 断开时清零 + 自动重稳定 (2026-06-25)
- [ ] left 模式示教 + t_cup_offset_left 微调
- [ ] free 模式完整姿态测试 (当前仅 vertical 模式做 B/C 修正)
- [ ] Orbbec / RealSense 新版集成
