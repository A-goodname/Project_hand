# 机械臂+灵巧手抓水瓶全流程

本文档将位姿转换拆分为两个阶段：先计算末端接触位姿（理论已验证可行），再叠加工具偏移得到抓取TCP位姿。

## 0. 坐标系与符号约定

| 符号 | 含义 |
|------|------|
| P_obj_cam | 水瓶在相机坐标系下的6D位姿（YOLO 6D检测输出） |
| T_cam_ee | 相机到末端法兰的4x4齐次变换矩阵（手眼标定结果） |
| P_ee_base | 当前末端法兰在基坐标系下的6D位姿（机械臂反馈，记为 [x,y,z,A,B,C]） |
| T_ee_base | 从末端法兰到基坐标系的4x4齐次矩阵（由 P_ee_base 构造） |
| t_tool_ee | 标准安装（C=0，大拇指向上）时，工具中心相对末端法兰的平移偏移 [0.06, -0.04, 0.07] m（经手动调优） |
| r_tool_ee | 旋转偏移，即灵巧手绕末端Z轴的安装角度，[0, 0, -90] 度（C=-90 时掌心向下，手指水平开合） |
| Rz(theta) | 绕Z轴旋转 theta 度的3x3旋转矩阵 |

## 第一部分：末端接触（已验证可行）

目标：计算机器人移动到某一姿态，使得末端法兰中心对准水瓶上的抓取点位置。

### Step 1：相机检测

YOLO 6D 检测得到水瓶在相机系下的位姿：
P_obj_cam = (x_cam, y_cam, z_cam, A_cam, B_cam, C_cam)

**物体坐标系约定 (第1步)**: PCA 输出后, Z 轴强制指向相机 (target_z[2] > 0 时翻转) . 此操作在 test_6D.py 的 compute_6d_pose 内部完成.

### Step 2：转换到末端法兰坐标系

P_obj_ee = T_cam_ee . P_obj_cam

展开为：
t_obj_ee = R_cam_ee . t_obj_cam + t_cam_ee
R_obj_ee = R_cam_ee . R_obj_cam

其中 T_cam_ee = [[R_cam_ee, t_cam_ee], [0, 1]]，得到 P_obj_ee = (x_ee, y_ee, z_ee, A_ee, B_ee, C_ee)。

## 第二部分：工具偏移（基于末端接触叠加）

目标：在末端接触的基础上，叠加灵巧手的物理偏移，使TCP（工具中心点）对准水瓶。

### Step 3：计算实际偏移量

灵巧手实际安装角度为 C=-90 度（由机械安装决定）。标准平移偏移需旋转到当前安装方向，用于位置补偿：
t_tool'_ee = Rz(-90) . [0.06, -0.04, 0.07]     (单位：m)

R_ee_base 由机械臂当前末端位姿构造（用于 Y 轴符号修正）。

### Step 4：计算目标TCP位姿（末端系下）

**位置补偿**：在末端坐标系下，工具中心需要到达的位置 = 物体位置 - 工具偏移
t_grasp_ee = t_obj_ee - t_tool'_ee

**方向对齐**：根据抓取模式计算法兰的目标旋转矩阵 R_grasp_ee

YOLO 6D输出的坐标约定：
- R_obj(:,0) = X轴 (Y x Z)
- R_obj(:,1) = Y轴（PCA主方向，瓶身轴线）
- R_obj(:,2) = Z轴（指向相机）

**物体坐标系约定 (第2步)**: 在 compute_grasp_rotation 中, 通过基系 Z 分量判断 Y 轴符号, 统一指向竖直向下, 消除 PCA 符号歧义. 此操作在 utils_shared.py 的 compute_grasp_rotation 第一步完成.

三种模式定义，均先统一水瓶 Y 轴指向为竖直向下（通过基系 Z 分量判断符号）：

| 模式 | 计算方式 | 说明 |
|------|---------|------|
| free | 统一 Y 向下后, R_grasp_ee = R_obj_ee @ Rz(+90) | 跟随水瓶朝向并叠加安装补偿，适用于非竖直/非水平放置。Rz(-90) 为安装偏移，取其逆 Rz(+90) 补偿。 |
| vertical | 在基系下约束末端 Z 轴水平（XY 平面 ∥ 基系 Z 轴），构造 R_grasp_base 后反推回末端系 | 从侧面抓取竖直水瓶，抓取姿态不受物体朝向数值巧合影响 |
| horizontal | 暂未实现 | — |

**vertical 模式详细算法**：
1. 将水瓶朝向转到基系：R_obj_base = R_ee_base @ R
2. 提取 y_obj_base（瓶身轴线，已统一向下）和 z_obj_base（指向相机）
3. 末端 Z 轴 = z_obj_base 在基系 XY 平面的水平投影（强制 Z 分量为 0）
4. 末端 X 轴 = y_obj_base（竖直向下）
5. 末端 Y 轴 = 右手系，由 Z x X 确定，再重新正交化确保精度
6. 组装基系目标旋转矩阵 R_grasp_base
7. 反推回末端系：R_grasp_ee = R_ee_base^T @ R_grasp_base

### Step 5：转换到基坐标系（最终机器人目标位姿）

将末端系下的TCP目标转换到基系：
R_ee_base = Rz(C_robot) @ Ry(B_robot) @ Rx(A_robot)
rot_base = R_ee_base . R_grasp_ee
t_grasp_base = R_ee_base . t_grasp_ee + t_ee_base

将 rot_base 拆解为 (A, B, C) 欧拉角。

### Step 6：可达性修正

将基坐标系下的 A 和 C 角加减 180 度，使目标位姿尽可能接近当前末端位姿，减少机器人关节运动距离。B 角为中间轴，其 ±180 度会改变末端朝向，因此不做修正。

得到最终发送给机器人的6D位姿：
P_target = (x_grasp_base, y_grasp_base, z_grasp_base, A_grasp_base, B_grasp_base, C_grasp_base)

## 完整转换链路

```
相机检测 ----> P_obj_cam
    | 手眼标定 T_cam_ee
    v
P_obj_ee ----> 减去工具偏移 t_tool'_ee ----> t_grasp_ee  (位置)
         ----> 方向对齐 + Y轴符号修正         R_grasp_ee (旋转)
    | T_ee_base
    v
P_grasp_base ----> 可达性修正(A/C±180°) ----> 发送机器人
```

## 关键说明

1. 第一部分是已验证的：不使用夹爪偏移时，末端法兰已能接触到水瓶，说明手眼标定和位姿转换链路正确。
2. 第二部分是增量：仅需在末端坐标系下对位姿进行平移补偿 (-t_tool'_ee) 和旋转补偿，无需重新进行标定。
3. 旋转补偿的优先级：先对位置做平移补偿，再考虑方向对齐。方向对齐只在物体不是竖直圆柱时才有实际意义。
4. PCA 主方向存在符号歧义（Y 轴可能向上或向下），通过将 Y 轴在基系下的 Z 分量统一为负值解决。
5. 参数 t_offset 经手动调优，当前值为 [0.06, -0.04, 0.07] m，可在需要时根据抓取效果微调。