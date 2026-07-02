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

### P3 示教

- [x] 示教脚本 `estun_teach_p3.py` 日志增强: 新增 `模式: back/left` 记录
- [x] back 模式 P3 示教数据标定 (4 次取均值)
- [x] `p3_teach_offset.back` 配置更新为 `[0.0126, -0.1361, 0.2127]` / `[-90.0, 30.3, -90.0]`
- [x] **back 示教点实际倒水精度验证通过**
- [x] P3 多倒水角度支持: `estun_config.yaml` 按角度索引 (10°/15°/20°/30°), `pour_angle` 配置项切换
- [x] **left 模式 P3 示教完成** (3 次取均值, 30°): `t_offset=[-0.0175, -0.0765, 0.2100]`
- [x] **left 模式 10°/15°/20° Demo 逐角度实测校正** (Y 偏移微调)
- [x] **left 模式全流程 Run 实测通过**
- [x] **Bug 修复**: PyYAML 整数键 vs `str()` 字符串键不匹配导致 pour_angle 回退到第一个 key (10°→误用 10° 而非配置的 30°), demo + run 双文件加 `{str(k): v for k, v in ...}` 统一转字符串

### 抓取微调

- [x] t_handle_offset 微调完成
- [x] t_cup_offset 微调完成 (back 模式)
- [x] t_cup_offset_left 微调完成 (left 模式)

### OpenCV 窗口修复

- [x] 窗口标题去中文

### 模型微调

- [x] 水壶检测模型微调 (KettleTraining/)
- [x] 数据采集 + 标注工具

### 灵巧手控制

- [x] USB 串口通信
- [x] 手势动作 (actlist.json)
- [x] 速度控制
- [x] 使能/关闭

### 通信协议

- [x] PC端 `estun_handle_cup_run.py` 重构: `fmt_pose()`/`pack_three()` 抽函数, 清零时机改为断开时, 自动重稳定流程
- [x] 机械臂端 `estun_handle_cup_code.txt` v3: flag条件循环 + P4预倒水点 + Cup完整流程(RELEASE/GRIP/RESET) + MovL Home双位置
- [x] 机械臂端 `.crp` (handlepluscup-Down-826.crp): 47控件, H_P3单位修正, Cup TranStrToCpos+运动序列+信号补全
- [x] PC端 Phase C: RELEASE/GRIP 指令处理
- [x] `()` → `<>`: `pack_three` P1 分隔符改为 `<>`
- [x] `_poses_cleared` UnboundLocalError 修复
- [x] `[Handle STABLE]` / `[Cup STABLE]` 稳定位姿输出
- [x] Git 署名清理

### 灵巧手可靠性

- [x] `init_hand()` 使能重试
- [x] `is_hand_available()` 函数

### 双相机体系

- [x] Step 1: Serial 绑定
- [x] `EyeToHand/` 眼在手外模块:
  - `person_detector.py`: `PersonDetector` + `PersonStabilizer`
  - `config.yaml`: 外置相机 + 预示教递送 + 稳定器参数
  - `detector.py`: `load_calibration()` / `camera_x_to_base_x()` 工具函数
  - `calibration/calibrate.py`: 1D最小二乘线性回归
  - person_detector 已测试通过; calibrate.py 未实操

### 工程

- [x] `.clinerules/project.md` — 项目级规则
- [x] `memory-bank/` — 上下文持久化
- [x] Conda 环境配置
- [x] YAML 配置管理

## 待完成

### 倒水流程

- [x] P4 预倒水点: 已在示教器测试通过 ✅ (2026-07-02)

### 双相机体系 — Orchestrator 状态机

- [x] **Step 5-7: `Orchestrator/` 状态机 + 总入口 + 递送位置计算** ✅ (2026-07-02)
  - `state_machine.py`: `OrchestratorState` 枚举 + `StateMachine` 引擎 + 6 个 handler
  - `main.py`: 双相机主循环 + 可视化 + 异常清理
  - `HandleCupRunner`: 封装 Handle+Cup 全流程 (复现 estun_handle_cup_run.py 核心逻辑)
  - 递送协议: 单 pose `<x,y,z,A,B,C>` + READY 确认 + RELEASE + 自主退回
- [ ] **`EyeToHand/calibration/calibrate.py` 实操标定** (P0, 当前唯一阻塞项, 需机械臂)
- [ ] **填入递送参数**: `EyeToHand/config.yaml` 的 `delivery.Y_TEACH/Z_TEACH/ABC` 预示教值
- [ ] **双相机体系 Step 8-10**: 联调 + 异常处理 + 全面测试

## 已知问题

- RealSense_R 旧相机的 rs.align 不可用 (硬件内参损坏), 需手动像素映射
- 部分脚本仍有类式灵巧手接口遗留 (已逐步迁移到函数式)
- `estun_handle_run.py` 通信协议未同步 handle-cup 版的 v2 改进 (暂缓)
- `EyeToHand/calibration/calibrate.py` 标定工具未实操验证 (机械臂被占用)
- `EyeToHand/calibration/result.yaml` scale/bias 为默认占位值 1.0/0.0, 需标定后更新
- **free 模式不应限制角度**: 当前 B/C 角约束已加 `if ORIENTATION_MODE == "vertical"` 守卫, free 模式仅做 post_correct_angles 可达性修复
