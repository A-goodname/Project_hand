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
- [x] `cup_grasp.pour_offset.r_offset` 从 -45° 更新为 -55°
- [x] 示教模式 P3 公式修正: 移除多余的 `+ POUR_T_OFFSET_BASE`
- [x] **back 示教点实际倒水精度验证通过**
- [x] P3 多倒水角度支持: `estun_config.yaml` 按角度索引 (10°/15°/20°/30°), `pour_angle` 配置项切换

### 抓取微调

- [x] t_handle_offset 微调完成 (用户自行在 estun_config.yaml 调整)
- [x] t_cup_offset 微调完成 (用户自行在 estun_config.yaml 调整)

### OpenCV 窗口修复

- [x] 窗口标题去中文: `"estun Handle+Cup Demo (RealSense 新模块)"` → `"estun Handle+Cup Demo (RealSense)"`

### 模型微调

- [x] 水壶检测模型微调 (KettleTraining/)
- [x] 数据采集 + 标注工具

### 灵巧手控制

- [x] USB 串口通信
- [x] 手势动作 (actlist.json)
- [x] 速度控制
- [x] 使能/关闭

### 通信协议

- [x] PC端 `estun_handle_cup_run.py` 重构: `fmt_pose()`/`pack_three()` 抽函数, 清零时机改为断开时, 自动重稳定流程 (经 code-review 审阅)
- [x] 机械臂端 `estun_handle_cup_code.txt` v3: flag条件循环 + P4预倒水点 + Cup完整流程(RELEASE/GRIP/RESET) + MovL Home双位置
- [x] 机械臂端 `.crp` (handlepluscup-Down-826.crp): 47控件, H_P3单位修正, Cup TranStrToCpos+运动序列+信号补全
- [x] PC端 Phase C: RELEASE/GRIP 指令处理 (`release_object()`/`grip_object()`)
- [x] `()` → `<>`: `pack_three` P1 分隔符改为 `<>` (机器人 `TranStrToCpos` 不支持 `()`)
- [x] `_poses_cleared` UnboundLocalError 修复: `main()` global 声明补全
- [x] 主循环去掉逐帧打印, 改为 `[Handle STABLE]` / `[Cup STABLE]` 稳定位姿输出
- [x] Git 署名清理: `~/.claude/settings.json` attribution 置空, 历史提交 Co-Authored-By 移除

### 灵巧手可靠性

- [x] **`init_hand()` 使能重试**: `enable()` 后 `is_enabled()` 验证硬件实际状态, 未确认自动重试 (10 次/0.5s)
- [x] **`is_hand_available()`**: 新函数替代 run 脚本 `_hand_available = True` 乐观标志
- [x] `estun_handle_cup_run.py` / `estun_handle_run.py` 导入并使用 `is_hand_available()`

### 双相机体系

- [x] Step 1: Serial 绑定 (RealSense + RealSense_R 各自 `config.enable_device()`)
- [x] `EyeToHand/` 眼在手外模块:
  - `person_detector.py`: `PersonDetector` + `PersonStabilizer` 稳定检测
  - `config.yaml`: 外置相机 + 预示教递送 + 稳定器参数
  - `detector.py`: `load_calibration()` / `camera_x_to_base_x()` 工具函数
  - `calibration/calibrate.py`: 1D最小二乘线性回归 (交互式, N≥2点)
  - person_detector 已测试通过; calibrate.py 未实操 (机械臂被占用)

### 工程 (Cline 配置)

- [x] `.clinerules/project.md` — 项目级规则 (Memory Bank + Git + 代码风格 + 模块 + 环境)
- [x] `Documents\Cline\Rules\context.md` — 全局 New Task 提醒策略 (300K/15轮)
- [x] `memory-bank/` — 上下文持久化 (6 个核心文件)
- [x] Conda 环境配置
- [x] YAML 配置管理
- [x] `doc/next_steps_prompt.md` 已读取并删除

## 待完成

- [x] P4 预倒水点: 机械臂端 CposOffset 实现 (doc/estun_handle_cup_code.txt v2)
- [x] RealSense (新版, rs.align) 集成到 estun_handle_cup_* (已完成, 无需 Orbbec 集成)
- [x] `doc/estun_handle_cup_code.txt` 通信超时 2s → 5s (2026-06-30)
- [x] `doc/Future_work.md` v3: 双相机体系架构设计 (简化递送策略 + 2点标定 + 砍手势识别) (2026-06-30)
- [ ] P4 预倒水点: 待在实际示教器上测试 (P0, 硬件被占用)
- [ ] **left 模式示教** + t_cup_offset_left 微调 + P3 多角度配置 (当前焦点)
- [x] **双相机体系 Step 2-4**: `EyeToHand/` 模块搭建 + 人体检测 + 1D最小二乘标定 (2026-06-30). person_detector 已测试通过; calibrate.py 代码完成, 未实操 (机械臂被占用)
- [ ] **标定验证**: `EyeToHand/calibration/calibrate.py` 实操 (P0, 机械臂被占用)
- [ ] **双相机体系 Step 5-7**: `Orchestrator/` 状态机 + 总入口 + 递送位置计算
- [ ] **双相机体系 Step 8-10**: 联调 + 异常处理 + 全面测试
- [ ] `estun_handle_run.py` 通信协议同步 (暂缓, 待现有协议验证后再做)
- [ ] 全面自动化测试

## 已知问题

- RealSense_R 旧相机的 rs.align 不可用 (硬件内参损坏), 需手动像素映射
- 部分脚本仍有类式灵巧手接口遗留 (已逐步迁移到函数式)
- 水杯 handle_side=left 模式的偏移量需实际标定 (P0 阻塞)
- `estun_handle_run.py` 通信协议未同步 handle-cup 版的 v2 改进 (暂缓)
- `EyeToHand/calibration/calibrate.py` 标定工具未实操验证 (机械臂被占用)
- `EyeToHand/calibration/result.yaml` scale/bias 为默认占位值 1.0/0.0, 需标定后更新
- **free 模式不应限制角度**: 当前 B/C 角约束已加 `if ORIENTATION_MODE == "vertical"` 守卫, free 模式仅做 post_correct_angles 可达性修复, 不修正角度
