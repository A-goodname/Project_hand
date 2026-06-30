# Active Context — Project_hand

## 当前焦点

**P2 双相机体系 Step 2-4 完成**: `EyeToHand/` 模块搭建完成 — 人体检测 (PersonDetector + PersonStabilizer) + 标定工具 (2点1D最小二乘) + 3D定位换算.
person_detector 已测试通过. calibrate.py 代码完成, 未实操验证 (机械臂被占用).

## 最近变更

1. **EyeToHand/ 眼在手外模块搭建** (2026-06-30):
   - 命名对齐项目 PascalCase 风格 (草案 `hand_to_eye/` → `EyeToHand/`)
   - `person_detector.py`: `PersonDetector` 直接继承 `BaseRealSenseDetector` (对齐 test_3D.py 模式), YOLOv8 COCO person class_id=0, 输出 bbox 中心 3D 相机坐标
   - `PersonStabilizer`: 窗口稳定器 (参考 PoseStabilizer), window=10/pos_thresh=0.03m (config.yaml 可配)
   - `detector.py`: 工具函数 `load_calibration()` / `camera_x_to_base_x()` (scale+bias 1D 映射)
   - `calibration/calibrate.py`: 2点1D标定 (最小二乘线性回归), 末端 YZ/ABC 保持示教值仅沿 X 移动, 终端交互输入 X_base
   - `calibration/result.yaml`: 标定输出模板
   - person_detector 已测试通过; calibrate.py 代码完成未实操 (机械臂被占用)

2. **P2 双相机体系架构设计** (2026-06-30):
   - `doc/Future_work.md` v2: 全面重写, 含硬件布局、serial 绑定方案、`hand_to_eye/` + `orchestrator/` 模块设计、6 状态状态机、开发计划 (Phase 1-3, 10 Step)
   - T4: `doc/estun_handle_cup_code.txt` 通信超时 2s → 5s (两处 `SocketReadStr`)
   - 明确 T7 (左把手抓取) vs T2+T3 (left 模式示教) 的区别

3. **机械臂端 .crp / 参考代码 / PC端 全面审查与修复** (2026-06-29):
   - 修复 H_P3 TranStrToCpos 单位 `mm` → `m`
   - 补全 Cup 阶段: 3 个 TranStrToCpos + 14 个 Cup 运动控件
   - While 循环改用 `strflag` 条件循环, 尾部补 Wait(500ms)
   - MovL Home 双位置 + Phase C RELEASE/GRIP 指令处理

4. **灵巧手使能可靠性修复** (2026-06-26):
   - **问题**: `OmniHandGesture.enable()` 返回 True 仅表示 ACK 收到, 不代表硬件真正使能.
   - **`Estun/utils_shared.py` — `init_hand()`**: `enable()` 后 `is_enabled()` 验证 + 自动重试 (10 次/0.5s)
   - **`is_hand_available()`**: 新函数替代 `_hand_available = True`

5. **通信协议与流程重构** (2026-06-25):
   - PC端: `fmt_pose()`/`pack_three()` 抽函数, 清零时机改为断开时, 自动重稳定流程
   - 机械臂端: `estun_handle_cup_code.txt` v2 — 线性流程 + P4 预倒水点

6. **back 模式 P3 示教验证通过**
7. **B/C 角防跳变修正** (2026-06-23)

## 下一步

- P4 预倒水点: 待在实际示教器上测试 (P0, 硬件被占用暂不可行)
- left 模式示教 + t_cup_offset_left 微调 (P0, 同上; left 为空时先用 back 值兜底)
- **标定验证**: calibrate.py 待机械臂可用后实操 (2点采集 + 拟合 + 保存)
- **Step 5-7**: `Orchestrator/` 状态机 + 总入口 + 递送位置计算
- 全面自动化测试
- `estun_handle_run.py` 通信协议同步 (暂缓, 待现有协议修改验证后再做)
- `estun_handle_run.py` 通信协议同步 (暂缓, 待现有协议修改验证后再做)
