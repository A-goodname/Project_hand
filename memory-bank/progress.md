# Progress — Project_hand

## 已实现功能

### 瓶身抓取

- [x] RealSense 实时版 (Estun/estun_run.py + RealSense_R/)
- [x] EpicEye 单次触发版 (Estun/estun_epiceye_run.py + EpicEye/)
- [x] 位姿转换链路验证 Demo (estun_demo.py / estun_epiceye_demo.py)
- [x] TCP 服务器模拟 (estun_test.py)
- [x] 手眼标定工具 (calib_data/)

### 把手抓取

- [x] RealSense Handle 6D 版 (Estun/estun_handle_run.py + RealSense_R/)
- [x] 微调 YOLOv8n + SAM2 检测流程
- [x] 扫描线分离把手
- [x] 红色胶带辅助检测
- [x] PoseStabilizer 多帧稳定
- [x] 代际计数器防污染
- [x] 把手位姿转换 Demo (estun_handle_demo.py)
- [x] Handle P1 B/C 角 vertical 模式守卫 + B 角 clamp [84,86]→85

### 水杯抓取

- [x] 水杯 6D 检测 (RealSense/test_cup_6D.py)
- [x] 杯口定位 + 倒水目标点
- [x] 双检测器共享 Pipeline (estun_handle_cup_demo.py / estun_handle_cup_run.py)
- [x] Cup P1 B/C 角 vertical 模式守卫 (back/left 分模式, B 角防跳变)

### Handle+Cup 融合

- [x] `estun_handle_cup_run.py` — 抓壶倒水→抓杯递杯 全流程 ✅
- [x] back/left 双模式实测通过 ✅
- [x] P3 多角度示教 (10°/15°/20°/30°) ✅
- [x] P4 预倒水点示教器验证 ✅

### 灵巧手控制

- [x] USB 串口通信 / 手势动作 / 速度控制 / 使能
- [x] `init_hand()` 使能重试
- [x] `is_hand_available()` 函数

### 通信协议

- [x] `pack_three` `<>[]{}` 三括号标记
- [x] 机械臂端 `.crp` + `estun_handle_cup_code.txt` v3
- [x] PC 端 `fmt_pose()`/`pack_three()` 抽函数
- [x] `_poses_cleared` 修复 / 稳定位姿输出

### 双相机体系

- [x] Serial 绑定: Camera A (`819612071851`, RealSense rs.align) + Camera B (`342222071364`, RealSense_R)
- [x] `EyeToHand/` 眼在手外模块:
  - `person_detector.py`: PersonDetector + PersonStabilizer ✅
  - `detector.py`: `load_calibration()` / `camera_x_to_base_x()` 工具 ✅
  - `calibration/calibrate.py`: 1D 最小二乘线性回归 (代码完成) ✅
  - `person_detector` 独立测试通过 ✅
  - `hand_in_cup.py`: YOLOv8m+YOLOv8n-Pose 人手-杯把手重合检测 ✅
    - 可独立测试: `python EyeToHand/hand_in_cup.py` ⚡
- [x] **Orchestrator 状态机重构** (2026-07-03) ✅:
  - `state_machine.py`: 2 状态 (IDLE ↔ HANDLE_GRASP_AND_POUR), 单 TCP Session 全流程
  - `main.py`: 双相机主循环 + Person/HandInCup 检测切换
  - Camera A 修正为 `RealSense/` (rs.align)
  - Cup P3 = (X_person, Y_TEACH, Z_TEACH, ABC) — EyeToHand 动态 X + config 预示教
  - HandInCup 判定 → TRUE(CONTINUE)/FALSE(ABORT) 双分支

### 工程

- [x] `memory-bank/` — 上下文持久化
- [x] Conda 环境配置 / YAML 配置管理
- [x] 模型微调 (KettleTraining/)
- [x] **机器人端代码 v4** — 伪代码 + CRP 示教器文件 + PC 响应格式同步 ✅

## 待完成

### ⚡ 需要实际测试 (硬件依赖)

| 任务 | 文件 | 说明 |
|------|------|------|
| 🔴 实操标定 | `EyeToHand/calibration/calibrate.py` | 需机械臂, 红色胶带, 末端沿 X 移动 3-5 点 |
| 🔴 填入递送参数 | `EyeToHand/config.yaml` → `delivery.*` | Y_TEACH/Z_TEACH/ABC 示教值 |
| 🔴 机器人端 v4 程序录入示教器 | `doc/estun_handle_cup_code_v4.txt` 或 `doc/handlepluscup-v4.crp` | 将 v4 CRP 导入 ESTUN 示教器, 并示教 C_P4/DCp2/DummyCpos |
| 🟡 HandInCup 无遮挡实测 ✅ | `EyeToHand/hand_in_cup.py` | 人手侧握杯把, 无灵巧手遮挡 — **已测试, 详见下方** |
| 🟡 HandInCup 有遮挡实测 | `EyeToHand/hand_in_cup.py` | **灵巧手遮挡杯把场景 — 尚未测试** |
| 🟢 Orchestrator 联调 | `Orchestrator/main.py` | 双相机 + 机器人 + 灵巧手全流程 |

### 📋 待完成 (非硬件)

- [ ] 机器人端 .crp 程序修改 (适配新协议指令)
- [x] HandInCup 模型下载 (`yolov8n-pose.pt` → `model/`) ✅ (两模型均已在 `model/`)
- [ ] HandInCup 阈值调优 (有/无遮挡实测后调整 `dist_factor` / `depth_tolerance`)

## 已知问题

- RealSense_R 旧相机的 rs.align 不可用 (硬件内参损坏), 需手动像素映射
- `EyeToHand/calibration/result.yaml` scale/bias 为默认占位值 1.0/0.0, 需标定后更新
- HandInCup 无遮挡已实测, 但 `dist_factor`/`depth_tolerance` 修正参数仍需调优; **有灵巧手遮挡场景未测试**
- Orchestrator 机器人端 .crp 尚未适配新指令 (READY/CONTINUE/ABORT)

### 🧪 HandInCup 实测记录

**无灵巧手遮挡场景** (人手直接侧握杯把):
- 实测发现: 侧握姿态下手腕关键点落在杯 bbox 扩展区之外, `dist_factor=0.8` × bbox 对角线不足以覆盖.
- 修正: 引入 `dist_factor` (距离阈值系数) 和 `depth_tolerance` (深度一致性容差, 默认 5cm) 作为可调参数, 扩大判定半径的同时用深度差过滤误报.
- 状态: 修正参数 **已实现但未实地验证** (`dist_factor=0.8` 和 `depth_tolerance=0.05` 为初始猜测值).
- 下一步: 重测无遮挡场景, 逐步调整 `dist_factor` 找到合适阈值; 注意 `dist_factor` 过大会引入误报, 需配合 `depth_tolerance` 使用.

**有灵巧手遮挡场景** (Omnihand 抓杯把时手指遮挡杯把区域):
- ❌ 尚未测试.
- 预期风险: 灵巧手手指遮挡杯 bbox 和手腕关键点, 可能降低检测置信度或导致漏检.
- 需等待实物条件具备后实测.
- `estun_handle_run.py` 通信协议未同步 handle-cup 版 v2 改进 (暂缓)
- free 模式不应限制角度: 已加 `if ORIENTATION_MODE == "vertical"` 守卫
