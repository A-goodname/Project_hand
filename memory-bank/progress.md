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
- [x] **RealSense_R t_handle_offset fine_tune** (2026-07-07):
  - RealSense `[0.06, -0.028, 0.094]` → RealSense_R `[0.048, -0.037, 0.074]` (164610)
  - fine_tune 反算改为 Cup 式增量法 (更简洁)
  - 历史 RealSense_R 值 `[0.04, -0.03, 0.080]` (f2e3579) 可供对比

### 水杯抓取
- [x] 水杯 6D 检测 (RealSense_R/test_cup_6D.py)
- [x] 杯口定位 + 倒水目标点
- [x] 双检测器共享 Pipeline (estun_handle_cup_demo.py / estun_handle_cup_run.py)
- [x] Cup P1 B/C 角 vertical 模式守卫 (back/left 分模式, B 角防跳变)
- [x] **RealSense_R t_cup_offset fine_tune** (2026-07-07):
  - back: `[0.048, -0.009, 0.098]` (163920)
  - left: `[0.040, -0.060, 0.080]` (161033)
- [x] estun_cup_demo.py: import RealSense/→RealSense_R/ + clamp 同步融合 Demo

### Handle+Cup 融合
- [x] `estun_handle_cup_run.py` — 抓壶倒水→抓杯递杯 全流程
- [x] back/left 双模式实测通过 (RealSense 版)
- [x] P3 多角度示教 (10°/15°/20°/30°)
- [x] P4 预倒水点示教器验证
- [x] `estun_handle_cup_demo.py` 切换至 RealSense_R/ + 相机参数拷贝修复
- [x] `estun_handle_cup_run.py` 切换至 RealSense_R/ + 相机参数拷贝修复
- [ ] **融合 Demo/Run Cup 检测修复** — start_pipeline=False 时 cup 无位姿, run 全流程阻塞 (进行中)

### 灵巧手控制
- [x] USB 串口通信 / 手势动作 / 速度控制 / 使能
- [x] `init_hand()` 使能重试
- [x] `is_hand_available()` 函数

### 通信协议
- [x] `pack_three` `<>[]{}` 三括号标记
- [x] 机械臂端 `.crp` + `estun_handle_cup_code.txt` v3
- [x] PC 端 `fmt_pose()`/`pack_three()` 抽函数
- [x] `_poses_cleared` 修复 / 稳定位姿输出
- [x] **机器人端代码 v4** — 伪代码 + CRP 示教器文件 + PC 响应格式同步

### 双相机体系
- [x] Serial 绑定: Camera A (`819612071851`, RealSense rs.align) + Camera B (`342222071364`, RealSense_R)
- [x] `EyeToHand/` 眼在手外模块:
  - `person_detector.py`: PersonDetector + PersonStabilizer ✅
  - `detector.py`: `load_calibration()` / `camera_x_to_base_x()` 工具 ✅
  - `calibration/calibrate.py`: 1D 最小二乘线性回归 (代码完成) ✅
  - `person_detector` 独立测试通过 ✅
  - `hand_in_cup.py`: YOLOv8m+YOLOv8n-Pose 人手-杯把手重合检测 ✅
- [x] **Orchestrator 状态机重构** (2026-07-03) ✅
- [x] **HandInCup 检测器** (2026-07-03) ✅

### 工程
- [x] `memory-bank/` — 上下文持久化
- [x] Conda 环境配置 / YAML 配置管理
- [x] 模型微调 (KettleTraining/)
- [x] 配置版本管理: RealSense 旧值注释保留，RealSense_R 新值标注来源

## 待完成

### ⚡ 需要实际测试 (硬件依赖)
| 任务 | 说明 |
|------|------|
| 🔴 修复融合 Demo Cup 检测 | start_pipeline=False 时 cup 无位姿 |
| 🔴 实操标定 `EyeToHand/calibration/calibrate.py` | 需机械臂 |
| 🔴 填入递送参数 `delivery.Y_TEACH/Z_TEACH/ABC` | 示教值 |
| 🔴 机器人端 v4 程序录入示教器 | 导入 CRP + 示教 C_P4/DCp2/DummyCpos |
| 🟡 Handle offset 多测验证 | 对比历史值 `[0.04, -0.03, 0.080]` |
| 🟡 Handle P3 示教验证 | p3_teach_offset 在 RealSense_R 下是否有效 |
| 🟡 HandInCup 无遮挡重测 | 调优 dist_factor/depth_tolerance |
| 🟡 HandInCup 有遮挡实测 | 灵巧手遮挡杯把场景 |
| 🟢 Orchestrator 联调 | 双相机 + 机器人 + 灵巧手全流程 |

### 📋 待完成 (非硬件)
- [ ] 融合 Demo/Run Cup 检测修复
- [ ] Handle offset 多测几组取均值
- [ ] P4 Y 后退 CRP 实现 (设计已定, 机械臂端未编码)
- [ ] HandInCup 阈值调优 (有/无遮挡实测后)
- [ ] Orchestrator 机器人端 .crp 适配新指令

## 已知问题
- RealSense_R 旧相机的 rs.align 不可用 (硬件内参损坏), 需手动像素映射
- `EyeToHand/calibration/result.yaml` scale/bias 为默认占位值 1.0/0.0
- HandInCup `dist_factor`/`depth_tolerance` 修正参数未实地验证
- **融合 Demo/Run 中 cup 检测 (start_pipeline=False) 无法获取位姿**，单独 Cup Demo (start_pipeline=True) 正常
- `estun_handle_run.py` 通信协议未同步 handle-cup 版 v2 改进 (暂缓)
