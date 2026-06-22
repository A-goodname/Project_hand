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

### 水杯抓取
- [x] 水杯 6D 检测 (RealSense/test_cup_6D.py)
- [x] 杯口定位 + 倒水目标点
- [x] 双检测器共享 Pipeline (estun_handle_cup_demo.py / estun_handle_cup_run.py)

### 模型微调
- [x] 水壶检测模型微调 (KettleTraining/)
- [x] 数据采集 + 标注工具

### 灵巧手控制
- [x] USB 串口通信
- [x] 手势动作 (actlist.json)
- [x] 速度控制
- [x] 使能/关闭

### 工程 (Cline 配置)
- [x] `.clinerules/project.md` — 项目级规则 (Memory Bank + Git + 代码风格 + 模块 + 环境)
- [x] `Documents\Cline\Rules\context.md` — 全局 New Task 提醒策略 (300K/15轮)
- [x] `memory-bank/` — 上下文持久化 (6 个核心文件)
- [x] Conda 环境配置
- [x] YAML 配置管理

## 待完成

- [ ] Orbbec 相机与主入口集成
- [ ] RealSense (新版, rs.align) 模块与主入口集成
- [ ] 双相机体系 (眼在手上 + 眼在手外) — 见 doc/Future_work.md
- [ ] 全面自动化测试

## 已知问题

- RealSense_R 旧相机的 rs.align 不可用 (硬件内参损坏), 需手动像素映射
- 部分脚本仍有类式灵巧手接口遗留 (已逐步迁移到函数式)
- 水杯 handle_side=left 模式的偏移量需实际标定