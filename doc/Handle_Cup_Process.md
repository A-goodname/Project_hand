# 水杯抓取与把手抓取融合 (Handle + Cup)

## 上次更新: 2026-06-11 (v3: 手势函数式重构)

---

## 1. 概述

将水壶把手抓取与水杯抓取融合，实现全流程自动化：

1. **Phase 1 — 抓水壶 + 倒水**: 检测水壶把手 → 抓取 → 移动到倒水位置 → 倒水
2. **Phase 2 — 抓水杯 + 递杯**: 检测水杯 → 抓取 → 移动到递杯位置 → 释放

共 6 个机器人位姿 (Handle P1/P2/P3_pour + Cup P1/P2/P3), 分两阶段发送.

---

## 2. 坐标系与符号约定

| 符号 | 含义 |
|------|------|
| P_obj_cam | 物体在相机坐标系下的 6D 位姿 (检测器输出) |
| T_cam_ee | 相机到末端法兰的 4x4 齐次变换矩阵 (手眼标定) |
| P_ee_base | 当前末端法兰在基坐标系下的位姿 [x,y,z,A,B,C] |
| T_ee_base | 从末端法兰到基坐标系的 4x4 齐次矩阵 |
| t_tool_ee / r_tool_ee | 工具相对末端的平移/旋转偏移 |
| Rz(theta) | 绕 Z 轴旋转 theta 度的 3x3 旋转矩阵 |

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

### 3.4 Handle P3_pour (倒水位置) — 关键变更点

P3 不再是简单的目标偏移, 而是计算到**壶嘴对准杯口**的位姿:

#### 输入:
- `pour_target`: 从 cup 检测器得到的倒水目标点 (杯口中心 + 径向偏移), 在相机坐标系
- `spout_offset` (`t_offset: [-0.06, 0.17, 0.0]`): 壶嘴在**抓取末端系下**的偏移
- `pour_offset` (`t: [0, 0, 0.05], r: [0, 0, -45]`): 基系 Z 轴防碰撞偏移 + 倒水旋转

#### 计算步骤:

**1. 倒水位置 → 相机系:**
```
pour_target_cam = pour_target  (已在相机系)
```

**2. 倒水位置 → 末端系:**
```
pos_pour_ee = T_cam2ee[:3,:3] @ pour_target_cam + T_cam2ee[:3,3]
```

**3. 反算抓取点末端位姿:**
壶嘴在抓取末端系下的偏移为 `spout_offset.t_offset`, 所以抓取点 = 壶嘴位置 - spout_offset (在抓取末端系下):
```
# spout_ee 是壶嘴在抓取末端系下的偏移 (几何约束)
# 倒水时, 壶嘴 = pour_pose, 所以抓取末端位置 = pour_pose - spout_offset
pos_ee_pour = pos_pour_ee - R_grasp_ee_corrected @ SPOUT_T_OFFSET
```

**4. 基系防碰撞偏移:**
```
pos_base_p3 = transform_pose_to_base(pos_ee_pour, robot_xyz, robot_ABC) + POUR_T_OFFSET_BASE
# POUR_T_OFFSET_BASE = [0, 0, 0.05]  (基系 Z 轴向上 5cm)
```

**5. 旋转:**
在 P1 修正姿态上叠加倒水旋转:
```
R_pour_ee = R_grasp_ee_corrected @ euler_to_rotation_matrix(*HANDLE_TARGET_R_OFFSET, order="zyx")
# 或直接用 POUR_R_OFFSET (目前同 HANDLE_TARGET_R_OFFSET = [0,0,-45])
rot_base_3 = R_ee_base @ R_pour_ee
```
解算 `(A3, B3, C3)` + 可达性修正.

---

## 4. Phase 2 — 水杯抓取 + 递杯

### 4.1 流程

```
(发送 'CUP' 标记)
MoveP2(cup) → RELEASE(PAPER) → MoveP1(cup) → GRIP → MoveP3(cup) → wait 1s(等人取)
→ MoveP1(cup) → RELEASE(PAPER) → MoveP2(cup) → RESET → MoveHome
```

### 4.2 Cup P1 (抓取点)

1. `RealSenseCup6DDetector.compute_cup_6d_pose()` 输出: `(bbox, position, rotation, euler, point_cloud, rim_pos, pour_target, rim_r)`
2. `position` 是水杯在相机系下的 3D 重心 (PCA 均值)
3. cam2ee 转换:
   ```
   pos_cup_ee = T_cam2ee[:3,:3] @ position + T_cam2ee[:3,3]
   rot_cup_ee = T_cam2ee[:3,:3] @ rotation
   ```
4. 位置补偿 (使用 `cup_grasp.gripper_offset`):
   ```
   t_cup_effective = R_z(r_cup_offset.C) @ t_cup_offset
   pos_ee_compensated = pos_cup_ee - t_cup_effective
   ```
5. 方向对齐 + 基系转换 (同 handle, 使用 `r_cup_offset` 和 `RIM_OFFSET_RATIO`)

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

基于 `estun_handle_run` 的 Phase C 指令循环扩展:

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

### 7.1 `estun_config.yaml` — cup_grasp 段

| 参数 | 值 | 说明 |
|------|-----|------|
| `enabled` | true | 总开关 |
| `spout_offset.t_offset` | [-0.06, 0.17, 0.0] | 壶嘴在抓取末端系下的偏移 (m), 已实际测量 |
| `gripper_offset.t_cup_offset` | [0.06, -0.04, 0.08] | 抓水杯的夹爪平移偏移 (m) |
| `gripper_offset.r_cup_offset` | [0, 0, -90] | 灵巧手安装角度 (度) |
| `pre_offset.t_offset` | [-0.0, 0.0, -0.05] | 预抓取平移偏移 (m) |
| `target_offset.t_offset` | [-0.2, 0, 0] | 递杯平移偏移 (m) |
| `target_offset.r_offset` | [0, 0, 0] | 递杯旋转偏移 (度) |
| `pour_offset.t_offset` | [0, 0, 0.05] | 基系 Z 轴防碰撞偏移 (m) |
| `pour_offset.r_offset` | [0, 0, -45] | 倒水旋转偏移 (度, 同 handle target) |

### 7.2 `RealSense_R/config.yaml` — cup 段

| 参数 | 值 | 说明 |
|------|-----|------|
| `height` | 0.100 | 杯高先验 (m), 实际测量 |
| `rim_radius` | 0.035 | 杯口半径先验 (m), 实际测量 |
| `rim_offset_ratio` | 0.6 | 倒水偏移比 (0~1) |

### 7.3 杯口方向确定

相机安装有一定角度 (非纯正面). 杯口方向通过**像素 Y 坐标**确定:

**流程 (Phase 3 PCA 后):**
1. 点云采集时同步记录每个点的像素坐标 (cx, cy)
2. PCA 得到主轴 `target_y_main`
3. 沿主轴投影: `proj = point_cloud @ target_y_main`
4. 取 `max_idx` (正方向端) 和 `min_idx` (负方向端)
5. 若正方向端在图像下方 (Y 像素大):
   ```python
   if pixels_arr[max_idx, 1] > pixels_arr[min_idx, 1]:
       target_y_main = -target_y_main
   ```
6. 主方向指数平滑: `target_y_main = 0.7 * last + 0.3 * current`, 再归一化

**关键修复:**
- 条件比对用 `max_idx` vs `min_idx` (而非反写)
- Phase 4 调用 `estimate_rim_position()` 时传入 `axis_in=target_y_main`, 避免函数内部重复 PCA 覆盖已修正的方向
- `estimate_rim_position()` 恢复使用 `center + axis * (cup_height / 2.0)` 计算杯口位置 (投影边界方法因俯视时点云离群点导致杯口偏出杯身, 已回退)
- 主轴指数平滑 (`AXIS_SMOOTH_ALPHA=0.7`) 抑制帧间 B 角跳动

---

## 8. 关键设计决策

### 8.1 倒水位置计算

壶嘴在抓取末端系下的偏移 (`spout_offset.t = [-0.06, 0.17, 0.0]`) 是在抓取时的末端坐标系下定义的.
倒水时, 壶嘴需对准杯口 pour_target, 因此需要通过 `spout_offset` 反算出抓取时的末端位姿.

### 8.2 Phase C 指令扩展

不改变原有指令循环结构, 仅添加 `CUP` 命令处理. 机械臂端在 handle 流程结束后发送 `CUP`, 本机回复 cup 三个位姿.

### 8.3 同步检测

Handle 和 Cup 检测使用同一帧数据 (在 Handle 检测帧时同步执行 Cup 检测), 避免两阶段检测的帧不一致问题.

### 8.4 灵巧手手势映射 (函数式重构)

2026-06-11 将 `HandController` 类 (位于 `estun_handle_run.py` / `estun_handle_cup_run.py` 中) 重构为 `utils_shared.py` 的函数式接口, 统一三套抓取流程:

| 函数 | 手势 | 适用场景 |
|------|------|----------|
| `init_hand(initial_gesture="PAPER")` | PAPER/RESET | 初始化, 水瓶默认 PAPER, 把手/融合默认 RESET |
| `hand_pre_grip()` | PREGRIP | 把手半闭合接近阶段 |
| `hand_grip_done()` | GRIPDONE | 把手完全夹紧 |
| `hand_reset()` | RESET | 复位灵巧手 (不关闭串口) |
| `grip_object()` | GRIP | 水瓶/水杯抓取 |
| `release_object()` | PAPER | 水瓶/水杯释放/预抓取张开 |
| `disable_hand()` | RESET→disable→close | 关闭灵巧手 |

移除 `estun_handle_run.py` 和 `estun_handle_cup_run.py` 中的 `HandController` 类及 `OmniHandGesture` 直接依赖, 统一从 `utils_shared` 导入.

### 8.5 当前状态

- `test_cup_6D.py`: 已验证通过, Y 轴方向固定, 杯口/倒水位置稳定
- `estun_handle_cup_demo.py / estun_handle_cup_run.py`: 已创建但未经实际相机运行验证, 下次开机后需测试双检测器同步及 6 点位姿输出
