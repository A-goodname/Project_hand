# Progress — Project_hand

### Phase 0 打招呼独立交付: py + CRP (2026-08-20)
- [x] 更新 doc/state_machine_reference.crp ← IDLEtemp-Down-645 (sha256 a8a8ee00, 状态机单独保存, 旧版 .bak)
- [x] data_sm-Down-946 与 doc/estun_handle_data_sm.crp sha256 相同 (4227c66b) — 已最新
- [x] 抽取独立 Phase 0 CRP: gen_greeting_phase0_crp.py → doc/estun_greeting_phase0.crp (sha256 d0dcdd5b, 26块/6点, Socket1:8600 四步)
- [x] 新建 Estun/estun_greeting_phase0.py: 独立 4 步 TCP, 人检测用迎宾 UGREEN usb-0:4.1, --mock 模拟
- [x] config greeting.camera_path: usb-0:4.1
- [x] 重新示教三个打招呼位姿 (2026-08-20): Z1/Z2/WS 已更新入 estun_config.yaml (CRP GH_* 零占位无需改)
- [x] py_compile + mock TCP 4 步回复全对 + 142 测试绿
- [ ] 实机验证: 示教器导入 estun_greeting_phase0.crp + PC 跑 estun_greeting_phase0.py 走通 4 步

### θ 稳定性记录工具 (2026-08-20, 新建脚本, 待实机)
- [x] 新建 `Orchestrator/tests/theta_stability_log.py` (import 复用 HandleCupDetector, 检测器本体未改)
- [x] 记录每帧 ratio/rd/θ/band/d/位置/姿态 + 累积 θ 均值/方差 ('r' 按姿势重置); 逐帧只写 log 不刷终端
- [x] 参数 `--skip` (默认 1=逐帧, %N==0 语义避开 %1==1 坑); **稳定窗口已去除** (先看原始 θ 数据)
- [x] 脚本整理: test_tag_points / mark_roi_hsv / theta_stability_log 移入 Orchestrator/tests/ (_project_root 2→3 层)
- [x] py_compile + import OK + 全量 142 测试绿 (unittest)
- [x] log 分析 (2026-08-20): band0 主信号近白噪声 σ0.5°, 5帧稳定窗口中位与全程均值 |偏差|≤0.4°、band 判对率 100%; band-15 单峰无抖档; band0 双峰 = 正侧 8° 接缝不连续 (非窗口问题, 待修)
- [x] config 定案 (2026-08-20): skip_frames 5→3 + stabilization.window 5→10 (保险; 4 SM 共享, cup/state_machine 亦变; 稳定耗时 0.83s→1s)
- [x] ROT_THETA_WIN 对齐 10 (2026-08-20, 用户定): 证据 sample 快照/theta_samples 5→10 样本, 18 回合证据合同随之变更 (采集前需训练侧知晓)
- [ ] 实机验证 10 帧窗口稳定行为; 训练侧申报证据合同 5→10

### pos_side v2: 双阈值 + 分段线性连续角度 (2026-08-20, 待实机)
- [x] v1 中间角度二值化问题: +15~+30 之间的实际摆位被强判 15/30, 正侧不连续
- [x] 物理限制确认: rd(θ) 在 +20° 后每度 <0.001 (噪声 0.002-0.003), +22~+30 不可分; +15 与 +20 rd 重合
- [x] 补 +20 数据 (144654 rd 0.3162), v2 json 现 24 点
- [x] pos_side v2: 双阈值 rd_a=0.314(15 上限)/rd_b=0.320(30 下限); 角度分段线性 (nodes 0/15/20/30°) + 30 档仿射
- [x] 效果: +20 摆位解算 19.96° (连续), 中间 rd 0.318→22.7°, +30 仿射 30.52, 判档 15/16
- [x] config pos_side 改 rd_a/rd_b/rd30/k/nodes; detector _interp_pos + 双阈值判档
- [x] 全量 135 测试全绿
- [ ] 实机只读验证各档 band/θ_med (尤其中间角度)
- [ ] 中间角度精度受 rd 噪声限制 (物理天花板, 需补中间摆位数据才更准)
- [ ] 异常点 110950 (d=0.531 近距) 重采确认

### v2 模型重拟合 + g 校正 (2026-08-20, g 校正已被 pos_side 替代)
- [x] `_detect_tag_region` 区域百分比化按 handle_side: 右侧 [70,110]% / 左侧 [-10,30]% (删 HANDLE_SIDE_RATIO)
- [x] clamp 图像边界 + 空区域 None; detect() 传 self._handle_side
- [x] test_tag_points.py 独立副本同步同区间
- [x] 新测试 6 例 + 全量 130 全绿
- [ ] 实机只读对比扩前后各朝向 ratio/θ/band (B2 后 +45 已工艺排除 θ∈[-35,+35], 正常摆位 ratio 回落更准; 确认不触发 0.34 REJECT)

### PoseStabilizer 合并 θ 稳定窗口 (2026-08-19, 待实机)
- [x] θ 并入 PoseStabilizer: 每帧 pos+euler+θ 同窗, 稳定判定含 θ 极差 (<5°), 直接取窗口 θ 中位数 → band; 删独立 θ deque + spread 手检查
- [x] config stabilization.window 3→5 (与 ROT_THETA_WIN 统一, B3 恰 5 帧; cup 窗口随之 5)
- [x] 删 _detect 双计数 (run 保留, frame_idx 每帧+1; 消除偶数 skip 陷阱)
- [x] B3 契约保持: 恰满窗/θ 分歧 REJECT/代际失效/禁 0° 回退/1.5s 新鲜度
- [x] 新测试 9 例 + 全量 101 全绿
- [ ] 实机只读验证: [Handle STABLE] θ_med/band + 稳定耗时 (3→5 帧 ~0.83s)

### Cup 阶段移植 + Camera B + Phase D (2026-08-18)
- [x] estun_teach_cup.py 新建 (Cup P1 示教工具, 镜像 teach_handle_p1)
- [x] cup_teach 示教验收: t_offset (-0.0226, 0.0021, -0.0078) m, 方差 <5mm ✓
- [x] estun_config.yaml 新增 cup_teach 段 (p1_offset + p1_abc + p3_pose)
- [x] estun_handle_cup_sm.py handle_teach + cup_teach 双链 (与 data_sm 对齐)
- [x] Camera B (usb-0:2) 接入 + PersonDetector + HandInCupDetector
- [x] camera.py path+name 联合匹配修复 (防 D435i 误匹配)
- [x] EyeToHand/config.yaml camera_path + cup_roi_fallback 重标
- [x] Phase D py 逻辑 (READY/RELEASE/CONTINUE/ABORT, 镜像 state_machine.py)
- [x] CRP Phase D blocks 添加 (TranStrToCpos + If 分支 + sudu2 + 新 points)
- [x] Phase A-C 实机验证通过 (Handle→倒水→Cup 抓取→递送)
- [ ] **Phase D 失败**: CRP 到 C_P3 后停止运动, 未发送 READY, 连接关闭 — 根因待查
- [x] P3 X+1cm 微调 (0.0686 → 0.0786 m)
- [x] doc 清理 (删除旧 session-barrier 文件, 更新 CRP)

### P1 示教工具 6D 偏置 (2026-08-16)
- [x] 稳定窗口收集姿态 (euler_cam, 只收 dynamic 帧) + 基系 6D 预测 (`transform_pose_to_base_6d`)
- [x] 偏置含角度: abc_offset = unwrap(P1_ABC - detect_ABC); yaml/log 输出补姿态
- [x] 离线验证: euler↔R 闭环 / 位置一致 / unwrap / mean_euler / 偏置闭环 全过
- [x] 实机适配: 非连续帧 (失败帧不清空 buf + SKIP=1) — 排坑 `%1==1` 恒 False bug; BUF_MAX_AGE 5s 防漂移
- [x] 姿态波动门限: 窗口内 euler 极差 <5° (对齐后) — 位置+姿态双门限才判稳定
- [x] 流程: R0 延后到稳定检测后 (首次必输/回车复用) + 示教阶段窗口关闭 (下次迭代重开)
- [x] 欧拉约定验证: A/B/C=绕基 X/Y/Z (Rz(C)Ry(B)Rx(A)), 绕竖直轴只改 C — 与用户实机观察一致
- [x] 实机示教跑通: 数据落盘 logs/teach_handle_p1_*.log (时间戳/相机系/极差/偏置/汇总统计, 不存 memory-bank)
- [x] 实机示教验收 (2026-08-17, HSV 链正对 4 位置): t_offset 方差 2.0/1.7/2.0mm ✓, abc_offset 0.2/2.5/2.0° ✓ (分析: logs/teach-p1-hsv-analysis-2026-08-17.md)

### P3 修正 + teach_p3 换源 (2026-08-17)
- [x] data_sm P3 姿态: 完整固定 ABC (p3_teach_r_abc, 与 P1 姿态不要求一致 — 用户约定; 曾试 AC 继承 P1 合成已回退)
- [x] estun_config.yaml 新增 home_pose (新 Home, 与示教器 CRP 同步)
- [x] teach_p3 检测器换 HandleCupDetector (与 data_sm 同源) + Home config 化 + 位置稳定 (输出格式未动)
- [x] 排坑: 方向键 ANSI 转义混入 input 丢示教 → teach_p3/teach_p1 加剥离
- [x] 重标 P3: 3 次示教 (剔除 cup z 异常次) → t_offset (0.0686,-0.0094,0.2536) + r_abc (-93.3,30.5,-5.3) → 已写 config left.30
- [x] t_offset 判定: P1 不用重标 (用户定案, handle_teach 保持 8-17 值)
- [x] data_sm 完整流程实机验证: P1/P2/P3/P4 全部正常 (抓壶→倒水→放回→Home 完美运行, 2026-08-17)
- [x] CRP 更新: data_sm-Down-638 → doc/estun_handle_data_sm.crp (38 widgets, 新增 H_P1 放回步骤, 旧版 .bak 备份)

### data_sm 接入 handle_teach (2026-08-17, 完成)
- [x] estun_config.yaml 新增 handle_teach 段 (参考 8-17 log: p1_offset/p1_abc, enabled: false)
- [x] data_sm `_load_configs` 加载 handle_teach + enabled 时 p1_abc 覆盖固定 ABC
- [x] Phase B P1/P2 位置双链切换 (enabled → P1=detect_base+t_offset 基系直接加 / P2=P1+R_base_fix@pre; 旧链对照保留)
- [x] 离线数学验证: log 4 次回放 P1 误差 0.7-3.2mm / P2 推导恒等 / 旋转恒等 R_ee_base@R_grasp_ee==R_base_fix
- [x] 实机只读跑: [Handle STABLE]/[Cup STABLE] 输出 + [TEACH] P1 落点对比 (正对场景) ✅
- [x] handle_teach.enabled: true → 完整流程低速验证通过 (2026-08-17, 抓→DCp1→倒水→放回→Home 全过)

### 波动根因修复: anchor 固定 seed + 系列回退 (2026-08-15)
- [x] `_get_anchor_depth` 固定 seed (确定性采样) — 实测 C 44°→3.9°、B 12°→2.2°、A 14°→8.9° (mask 稳定但姿态飘的根因)
- [x] mask 扩展默认关闭 (extend_px: 0); 中心回退 mask_center; 几何法回退 b0b36cd (Y=PCA 主轴)
- [x] 位置 x/y/z 1-2mm ✅
- [ ] A 角残余 8.9° (壶身轴 λ1/λ2=1.14 深度噪声) — 可选 rot_smooth_alpha 或壶身轴稳定化
- [ ] 实机示教 P1 (estun_teach_handle_p1.py) 待跑

### 姿态稳定性: Y 改用壶身轴×环法线 (2026-08-15)
- [x] `_pca_axis` 返回最小方差方向 (axis_min); 几何法 Y=cross(壶身轴, 环法线) — 由稳定轴决定, 不受弧面主轴摆动影响
- [x] fallback: 退化时 Y=PCA 主轴 (旧法, 保连续)
- [x] 中心回退 mask_center (点云重心); edge_offset 暂弃 (固定偏移相对 mask 过大)
- [x] 离线验证: 水平/符号/翻转一致/单测 11 例全绿
- [ ] 实机复测: C 角波动 (目标 <10°, 之前 48°)

### 检测器重构: 消除不符代码 + 边界偏移中心 (2026-08-15)
- [x] `_compute_handle_6d_pose` 简化为返回 (pos, y_axis, cloud) 主成分方向 (行为等价)
- [x] ~~`_extend_handle_mask` 判定修复 (body 中位数)~~ **已回退** — 实机 mask 完全不可用, 根因未明, 恢复 bbox 判定版
- [x] 文件头 docstring 更新 (几何法 + 边界偏移描述)
- [x] 边界偏移中心法 `_compute_handle_center_edge` (上边界向下 4cm + 把手侧边界向内 1cm, config 可切)
- [x] 离线验证: 编译 + 单测 11 例全绿
- [ ] 实机: 边界偏移中心效果 (用户验证中) + D1-A Task 4 复测

### 姿态 3D 几何法重构 (用户方案, 2026-08-15)
- [x] 移除 dx/θ/dx_ref/R_ref 自标定链 → `_compute_handle_pose_geometry` (X=壶身长轴/Y=把手长轴 符号按 grasp_direction+图像边界/Z=X×Y, **Z 朝向相机 = z<0** RealSense 约定)
- [x] scanline 严格化 (0.25 双向剔单行超差) + `_extend_handle_mask` **贪心扩展** (逐行补至连续 5 行无把手, 上限 50×4; 中心修正 y 183→226/240)
- [x] `_align_axes` 四候选帧间锚定 (恒等+绕 X/Y/Z 180°, 覆盖 Y/Z/X 全翻转模式)
- [x] 离线验证: Z 朝相机/4 模式锚定/贪心扩展全覆盖/单测 11 例全绿
- [ ] 实机复测 D1-A Task 4: 姿态抖动 <10°, Y 符号随 grasp_direction 正确, mask 扩展完整 (中心点不偏), Z 朝向相机

### 抓取流程细化: 180° 翻转修复 + P1 示教工具 (2026-08-15)
- [x] 180° euler 翻转修复: _align_y_axis 帧锚定 + 统一合成路径 + _MatrixSmoother EMA + _unwrap_euler + B 角监控 (3 层根因, 离线测试全过)
- [x] `Estun/estun_teach_handle_p1.py`: P1 示教工具 (基系偏置 t_offset = P1_teach - detect_base, 多点平均+方差, --apply 写 config, 变换链闭环验证)
- [x] 离线验证: py_compile + 单测 11 例全绿 + 翻转修复/闭环测试全过
- [ ] 实机复测: D1-A Task 4 场景 — 姿态抖动 <10° (目标 <5°)、A/B 无符号翻转聚类 (Step 1)
- [ ] 实机示教: 采集 ≥3 点位, 方差 <5mm, 输出 yaml 片段 (Step 2)
- [ ] **Step 3 (state_machine teach 模式) 暂停** — 用户指示, 待实机验证 Step 1/2 后恢复

### Handle 检测 HSV 化: SAM 链 → 青色贴纸链 (2026-08-17, 正对流程完成)
- [x] 新链: YOLO bbox → 右 30% HSV → 最大域 → 质心=中心 + 长短轴=6D 姿态 (X=短轴/Y=长轴/Z=X×Y)
- [x] 深度排坑: color_to_depth_pixel 坐标映射 (FOV 不同) + 质心邻域/区域内中位数 (不取背景)
- [x] HSV 参数标定: H 40-110 (贴纸青绿渐变 36-105), S/V ≥50
- [x] 移除 SAM 加载; 单测 31 例全过; 实机验证 (test_tag_points.py): 中心 ±0.2px / euler 0.1-0.2° / 深度 0.63m
- [x] P1 示教 (HSV 链, 正对 4 位置) 验收全过 (t_offset 2mm / abc_offset 2.5°) + 分析发训练侧 (teach-p1-hsv-analysis-2026-08-17.md)
- [x] 旋转实验: 2D fitEllipse 对绕竖直轴旋转不敏感 (±30° → C 仅 ~3°); 3D 点云 PCA 尝试 (法线噪声大) 已回退 — **A 角残余 8.9° 为旧点云 PCA 链问题，HSV 链不涉及**
- [x] 方案 A 宽高比法 (test_tag_points.py): ratio/theta 输出 + c 键标定 r0; r0=0.274 已标定 (±0.001)
- [x] 方案 A 验证 (2026-08-18): ±45° 内有效, 60°+ 遮挡失效; r 随位置/深度变化 (镜头畸变+透视) → **直接拟合 cos+sin+dd 回归更可靠** (物理偏角 α 模型弃用)
- [x] 方向不对称: 顺时针 vs 逆时针响应不对称 (水壶面向桌子长边/XZ 平面, 初始偏角 α≈28°) — cos+sin 的 sinθ 项捕捉
- [x] **工作范围 [-30°, +30°] 模型定版 (25 点重拟合, 2026-08-18)**: `ratio = 0.1147 + 0.1642·cosθ + 0.1027·sinθ - 0.0725·(d-0.63)`
  - 25 点拟合 (剔 pos30_3 离群 + 45 档另算), R²=0.9911, LOOCV MAE=0.0036, 角MAE=2.60° / max=11.48° (+30°端)
  - 与旧 18 点质量持平 (LOOCV 0.0037→0.0036 略好; 角MAE 2.34→2.60 略升因新增 +15 档本就弱)
  - **加 u/v 特征无实质提升** (u: LOOCV 0.0035 但角MAE 2.64; v: 17 点子集 R² 增量 0.0000) — cos+sin+dd 已最优
  - 判档 = **档内位置拟合 r̂_a(u,d) per-band** (JSON band_coeffs), LOOCV 88% (22/25), 0↔30 零混淆, 错误全在 15° 档相邻混淆
  - **0 vs +45 单独门限**: `u≥530 且 ratio≥0.2925` 判 +45° (margin +0.0129, 4/4+11/11 全对); u<530 不认 45 (u490 两点物理不可分, 实际不出现)
  - 反解: θ = φ - arccos((ratio-c0-c3·(d-d_ref))/A), φ=32.02°, A=0.1936; 标定数据 logs/ratio_angle_calibration.json
- [x] **抓取姿态规律 (2026-08-18)**: 绕竖直轴旋转 θ 只改 C, A/B 恒定 → `C = C0 - θ` (θ 顺时针为正)
  - 实测: 0° C=-21.1 → -15° C=-6.276 (差 0.176°) → -30° C=+8.919 (差 0.019°); A 偏差 ≤0.37° / B ≤0.04°
  - **姿态补偿只需转 C, 无需每档预示教姿态**; 位置检测自动跟随 (detect_base 变) — **P1 = detect_base + p1_offset 固定偏移不成立** (2026-08-18 用户纠正: T-Offset 需随 band 调整, 见下)
  - 记入 logs/ratio_angle_calibration.json `grasp_pose_calib` 段
- [x] **T-Offset 需随 band 调整 (2026-08-18, 用户指出)**: 旋转档位下固定 `p1_offset` (0° 示教值) 不再对准抓取点, 位置偏移方向随把手旋转 — 位置补偿不只是 detect_base 自动跟随, T-Offset 也要按 band 调整
- [x] **T-Offset 方案定案 (2026-08-19, 用户)**: 每档抓取**姿态 ABC 不再示教** (参考 C=C0-θ 规律, A/B 恒定); **每档 t_offset 需重新示教** (新档位)
  - 实机验证: -30 档可行; +45→+30 可容许 (45 判 30 可接受, 与离线结论一致)
  - 修正范围: 推翻 "P1 = detect_base + p1_offset 仍适用" 的旧推论; 待办 更新 data_sm P1 计算
- [x] **位置(uv)影响分析 (2026-08-18)**: r 随 u 正相关/随 d 负相关; 档内展宽 0.013~0.014 > 档间间隙 0.009 (0/15/30 档) → **纯 r 阈值必然错分, 需位置修正**
  - 0° 档 9 点位置线性拟合 R²=0.32 (仅 1/3 展宽是位置, 余为噪声/角度不精确); 30° 档位置敏感度是 0° 档 2 倍
  - **判档方案改为档内位置拟合**: 每档 a∈{0,±15,±30,±45} 拟合 r̂_a(u,d), 判档 argmin|r−r̂_a| (距离>阈值→无法识别)
  - 待补示教: +15(1→3-5位置), +30(3→+2-3), +45(重采3-5, 位置关系显著), -15(0→3-5), -30(可选)
- [x] **旋转检测集成进 handle_cup_detector.py** (2026-08-18): 用户定 6 档规则 (+45/+30/+15/0/-15/-30), 正半侧 per-band r̂_a(u,d) 最近邻 (45 需 u≥530), 负半侧反解 θ 区间; config.yaml handle_rotation 段; handle_result 新增 ratio/theta_deg/band; 离线验证 28/29 判对 + 29/29 可接受 (45→30 可接受, 无 45→15)
- [x] **band 实机验证 (2026-08-19)**: 0° ✅ 全对 (ratio 0.279 落标定中心) / +15° ⚠️ 94% 判 15 (6% 边界抖落 0) / **-15° ❌ 76% 判 0** (反解低估 ~5° + 零数据) / **+45° ❌ 判 15** (45 per-band 过拟合 + 遮挡致 ratio 0.294 与 +15 档 0.293 物理重合, 无一帧判 45)
- [x] **方向定案 (用户 2026-08-19)**: 补 -15 数据 → 拟合 **-30~+30 主模型** → **+45 用 u/v/d 等参数单独区分**
- [x] JSON 补 +45 实机点 (pos45_093515_实机, ratio0.2944), 现 33 点; config 未动
- [x] **补 -15 档数据 (2026-08-19)**: 3 个实机 log 提炼入档 (neg15_094705/094913/095007, ratio 0.2567/0.2650/0.2536, std≈0.001) → JSON 36 点
- [x] **重拟合 -30~+30 主模型 (2026-08-19, 28点)**: `ratio = 0.1071 + 0.1739·cosθ + 0.0988·sinθ - 0.0768·(d-0.63)`; R²=0.9780, LOOCV 0.0046; config global_model + JSON model 已同步
  - 反解偏差: -30 [+0.3,-2.0] / -15 [+3.4,+6.6] (系统性低估~5°) / 0 [-4.8,+2.1] / +15 [-5.3,+1.6] / +30 [-11.1,-0.4] (正侧饱和)
  - **三方案判档对比**: A现状代码 26/28, B档中心最近邻 27/28, C全档per-band 27/28 — 误判全为相邻档
  - **用户新标准 (2026-08-19)**: 相邻档误判可接受 (如 -15→0/-30 均可), 只需无跨档 → 三方案全达标, 撤销此前 -15→0 标记
  - **唯一硬伤 = +45→15 跨档** (全局模型 45° 档 ratio 预测与 +15/+30 几乎重合, 纯 ratio 不可分) → u/v/d 单独区分
- [x] **+45 档单独区分定案 (2026-08-19)**: u>586 高角度区 → 候选只留 {30,45} (config `u_high_zone: 586` + `_classify_rotation`); 补入裁切点 pos45_101155/102457; pos45_093515 标 excluded; 全量 35 点复验 +45 7/7 达标 (4判45/3判30), 真实跨档 0
  - 关键发现: +45 贴纸被 YOLO 框裁是稳定特征 (ratio 虚高 0.34~0.36 才能与 +15 分开); ROI 纵向扩展已回退 (裁切保留); u≥587 只有 +45 出现
- [x] **旋转判档简化定版 (2026-08-19, 用户)**: 只留 **5 档 -30/-15/0/+15/+30** (无 ±45), +45 统一落 +30; **纯 θ 区间** [-8,8]=0, (8,22]=15, >22=30 (负侧对称); 删 per-band/45 门限/u_high; config `band0_halfwidth=8` + `band15_hi=22`; `_classify_rotation` 重写 5 分支; JSON model 段同步
  - 校准 35 点: 精确 26 + 相邻 7 + 2 低u+45→15 (非操作); -30/-15/0/+15 全对, +30 3/5→30+2/5→15
  - 实机: +45 (u>560, ratio 0.34+) → 30 ✓; 干净 +30 (θ~16) → 15 band (5 档按 θ 映射)
- [x] **逐档示教 t_offset 完成 (2026-08-19, 5 档)**: p1_offsets 表 (-30/-15/0/+15/+30) 全齐; C=C0-θ 全档验证 (误差≤0.2°); X 单调 (≈-1.7mm/°); -15 预测 vs 实测 C 吻合 (X 偏9mm 因负侧非线性)
- [x] **data_sm band 查表 t_offset (2026-08-19, task #4 纯查表版)**: θ 5 帧中位数 → band → p1_offsets[band] (p1_offset + p1_abc); `_resolve_rot_band`/`_resolve_p1_offsets` 新增; Phase B P1 用 band 查表, 旧链回退固定 ABC
  - 子 agent 分析: X 对称线性 -1.72mm/°, Y 不对称弯曲 (线性 R²=0.82 失败), C 严格线性 C=C0-θ (R²=0.99998), Z 小量
  - 后续增强 (已实装 2026-08-19): **θ 档内线性插值** (`_interp_p1_offset`) + **C 解析式** (C=C0-θ); Z 钳位范围收紧 [0.360,0.362]
- [ ] 实机验证 data_sm (θ 插值 + C 解析式 + Z 钳位): 各档 P1 落点对比 → +15 档 Z 天然 364mm 超限观察钳位效果
- [ ] 多朝向示教: 旋转检测就绪后，各朝向（±30°/±60°等）示教 → abc_offset/detect_C 变化曲线验证

### Handle 检测回退 yolo+sam + 动态姿态 (2026-08-15)
- [x] `handle_cup_detector.py` 重构: YOLO kettle → SAM → 后处理 → scanline v2 → PCA 6D → 完整动态 euler_cam
- [x] mask 后处理: OPEN→ERODE→CLOSE→连通域过滤 (含把手侧孤岛保留)
- [x] scanline v2: mask 外接矩形搜索范围 + 保守剔除 (修复 bbox 截断/弧形把手被切)
- [x] 旋转角 θ: dx 特征 + dx_ref 自标定 + R_ref 正对参考缓存 + 壶身轴合成
- [x] config.yaml: sam/mask_postprocess/scanline/handle_ref/pca_step 新键, 删 white/red 段
- [x] 离线验证: py_compile + 单测 11 例全绿 + 合成 mask/数学测试
- [ ] 实机验证 (部分): 60/60 检测成功、位置抖动 0.7mm、P2 观测 0.46s ✅; **发现动态 euler 帧间 180° 翻转震荡 (A 轴 -100~-178/+130~+169 跳变, 位置稳定姿态不稳 ~100°) — PCA 主轴 180° 跳变未完全解决, 需修复后复测** (2026-08-15 D1-A Task 4)
- [ ] 后续: state_machine/estun SM 消费动态 euler (本次范围外)

### P2/P1 双确认协议适配 (2026-08-14, 进行中)
- [x] 交接文档: `doc/p2p1-dual-confirm-handoff-2026-08-14.md` + memory-bank 同步
- [x] 回传: `git diff 132051c..HEAD` (源码/requirements/配置) — 已发送对方 inbox 校验通过 (sha256 `30d64559...`)
- [x] 回传: CRP 文本版源码 — 离线解析版已发送对方 inbox 校验通过 (sha256 `45e8f571...`); 示教器官方导出版如需则待现场操作
- [x] 已发送: estun-omnihand-robotics skill 包 (tar.gz+sha256) → 对方 inbox 校验通过 (2026-08-14)
- [x] 回传: 运行入口清单 — `~/p2p1-handoff-staging/run-entry-inventory-2026-08-14.md` 已发送对方校验通过 (main.py/state_machine.py/estun_config.yaml/crp SHA-256 + 基准 commit 132051c→HEAD 2dd6fc9, 工作树 clean; 注: CRP 不受 git 管理, 版本以 SHA+mtime 为准)
- [ ] D0: 统一标定共享加载器, 失败阻止启动 (清 15 处 np.eye(4) 回退)
- [ ] D0: PREGRIP 与位姿确认分离
- [ ] P2/P1 双确认: CRP Phase A-C 消息序列改造 (P2_ARRIVED/GRASP_AUTH/P1_ARRIVED/GRIP_AUTH/GRIPDONE)
- [ ] 等对方基于实际 CRP 出精确补丁 → 故障注入回放 → D1 实机

### Linux 环境迁移适配 (2026-08-14)
- [x] `EyeToHand/camera.py` 跨平台: Linux V4L2 + MJPG + sysfs 设备名探测, Windows DSHOW 保留
- [x] Proj 环境依赖安装 (opencv/pyyaml/pyserial/pyrealsense2/ultralytics/mediapipe), 排坑: pyrealsense2 版本后缀 + opencv-contrib 版本冲突
- [x] 清理 __pycache__/.pyc + Windows 专属二进制 (win .pyd, RealSense SDK 安装器)
- [x] git 换行归一化: core.autocrlf input (86 假变更→3 真实) + .gitattributes (text=auto)
- [x] 灵巧手串口: by-id 稳定路径 + dialout 授权 + 只读身份 + RESET 自检 (O10, B655****0014)
- [x] ESTUN 握手: 4 步 CRP (连接/推流/'OK' 接收/关闭) 全通过, 机械臂 192.168.101.100
- [x] 三相机同时采集: UGREEN x2 (27/28fps) + D435i (28fps) 全速无冲突
- [x] 排坑: /dev/videoN 编号随插拔/驱动重绑漂移 → 必须按 sysfs 名称枚举, 禁止硬编码索引
- [x] `multi_camera_preview.py` 新建 (名称枚举版三相机预览)
- [x] 花屏排坑: D435i 与 UGREEN 同 hub 时等时带宽饱和 (video0 720p30 丢包花屏, 内核无报错); 把 D435i 挪到独立控制器后三相机 720p30 全速 0 丢包
- [x] 重登会话后验证 dialout 直连 (2026-08-16: `id` 确认已在 dialout 组, 直连无需 sg 包裹) ✅
- [x] 灵巧手串口 Linux 适配 (2026-08-16): 11 文件 31 处 COM5/COM3 → by-id 稳定路径 (voice_*/gesture/speed_control/test/usb_serial/state_machine/estun_handle_*_sm)

### 豆包端到端语音大模型 (全双工) 语音控制 (2026-08-13)
- [x] `Omnihand/voice_doubao_control.py`: 单文件 S2S 实现 (WS 会话 + 流式音频 + FC + 重连 + barge-in)
- [x] `voice_llm_prompt.py` 追加 DOUBAO_INSTRUCTIONS/DOUBAO_TOOLS/validate_tool_call/parse_fc_items (MiMo 部分未动)
- [x] `voice_llm_config.json` 追加 doubao 段 (api_key 空占位)
- [x] `requirements.txt` +websocket-client==1.8.0
- [x] mock 服务器 + 18 用例离线测试全通过 (FC 配对/barge-in/重连/优雅关闭)
- [x] 回归验证: MiMo 链路导入无异常, 两链路共存
- [x] `doc/voice_control_design.md` 附录 C
- [x] 豆包 API Key 申请 (控制台) + 写入 .local.json
- [x] 真实 API 免麦克风探测: WS/ASR/LLM/TTS/FC/优雅关闭 全链路 ✅
- [x] FC 排坑: tools 必须扁平格式 (OpenAI 包装不触发) — 已修 + 真实 API 验证
- [x] 排坑: 全双工必须持续推流 (52000033) — 主程序常开推流天然规避
- [x] 排坑: audio.delta 包大小不固定 → StreamingPlayer 跨块余量缓冲
- [x] 实机根因修复: mic_q 断链 (麦克风数据从未上送) → send 循环 _drain_mic 持续推流
- [x] 实机阶段 1 (部分): "你好" → ASR+FC+LLM 回复全链路 ✅ (扬声器播放/延迟对比待专项)
- [ ] 实机阶段 1 完整: 语音回复播放确认 + 首包延迟记录
- [ ] 实机阶段 2: 空载手势/速度三档/序列
- [ ] 实机阶段 3: 握手/急停/抢话/断网/音色
- [ ] MiMo vs 豆包延迟对比 (各 5 次)

### VLA 数据采集版 + CRP 完善 (2026-07-31)
- [x] `Orchestrator/estun_handle_data_sm.py`: Handle+Cup 检测 → P3 倒水点, 无 Cup 阶段
- [x] `doc/estun_handle_data_sm.crp` (35 widgets): 含 H_P4 回退避碰点 (CposOffset 动态计算), 无 Cup 段
- [x] `doc/estun_handle_cup_sm.crp` 补 H_P4 (53 widgets): Handle 倒水避碰, Cup 段保留
- [x] `doc/estun_handle_sm.crp` P3 速度 sudu → V100 (Z-up 抬升无需低速)
- [x] 窗口标题 em dash → ASCII `|` (双窗口/乱码修复)
- [x] 生成脚本参数化: `scripts_gen/gen_handle_data_crp.py --keep-cup`
- [ ] CRP 实机导入 + SM 联调

### 单相机状态机重构 (2026-07-31)
- [x] `Orchestrator/estun_handle_sm.py`: 仅水壶抓取, HandleCupDetector + 固定ABC + P3=Z-up
- [x] `Orchestrator/estun_handle_cup_sm.py`: 水壶+水杯, 同架构, Cup P3=Z-up
- [x] `doc/estun_handle_sm.crp` (23 widgets): 对应仅水壶 CRP
- [x] `doc/estun_handle_cup_sm.crp` (41 widgets): 对应水壶+水杯 CRP
- [x] `Orchestrator/config.yaml`: 新增 `handle_lift_z` + `cup_lift_z`
- [x] `handle_cup_detector.py`: 红色优先检测 (暗光适配)
- [x] 白色把手两遍检测 (紧→松)
- [ ] CRP 实机导入 + SM 联调

### Camera B 剪刀石头布手势识别 (2026-07-20)
- [x] 方案调研: YOLOv8n (OpenI/PCL) vs CNN vs MediaPipe几何 → 选定 MediaPipe 几何特征
- [x] YOLOv8n 预训练模型测试: 实机效果差, 已删除
- [x] MediaPipe 几何特征实现: PIP 关节 3D 角度 + 距离比值, 尺度不变
- [x] 核心 bug 修复: 角度阈值反了 (`>=150°` → `<=35°`)
- [x] 多人多手: num_hands=4, 每手独立 stable 跟踪 (4 帧确认)
- [x] 'q' 退出卡住: 匹配 hand_in_cup 模式 (独立 camera + waitKey)
- [x] clearcut 退出阻塞: GLOG_logtostderr=1 + os._exit(0), 全项目 3 处统一
- [x] 实机测试: 合适角度识别效果良好
- [ ] 集成到 Orchestrator 4 阶段调度
- [ ] 加水实测完整流程

### LLM 语义理解 (2026-07-17)
- [x] MiMo v2.5: speech+text-in text-out, 语义理解替代关键词匹配
- [x] MiMo TTS: 冰糖音色, 0.8s API 合成 WAV, 与 LLM 同 endpoint; 合成阻塞手势并发
- [x] System prompt: ~1200 chars, 9 examples, 社交交互自主配合手势
- [x] 双模式: auto (Silero VAD 默认) + ptt (按键)
- [x] 推理门: LLM+TTS 期间暂停 VAD 收指令
- [x] 音频: 16kHz VAD → 8kHz API → VAD 静音裁剪
- [x] 配置: voice_llm_config.json + .local.json (key gitignored)
- [x] 日志: logs/voice_llm_*.log
- [ ] LLM 语音控制实机联调

### 语音控制先行验证完成 (2026-07-17)
- [x] 全量模型切换: `model.onnx` (887MB), auto 优先选取
- [x] Silero VAD: `VoiceActivityDetector` + `silero_vad.onnx` 替换能量阈值
- [x] 5 层匹配流水线: 全局→速度→精确→字符模糊→拼音模糊
- [x] 拼音匹配: pypinyin TONE3 + SequenceMatcher, 自动处理同音字 (阈值 0.75)
- [x] 字符闸门: 汉字重叠度 ≥0.25, 防纯拼音碰撞
- [x] JSON 配置: `voice_commands.json` + 硬编码回退
- [x] 同义扩展: 新增 "招手"/"数字"/"数数" 等触发词
- [x] 防误触: 单字精确匹配 + 最长优先 + "握手"→序列(含压力检测)
- [x] 数字手势序列 delay: 0.5s → 0.8s
- [x] 模型目录重命名: `sherpa-onnx-sense-voice` (去日期后缀)
- [x] 去临时 WAV: transcribe() 直接用 accept_waveform
- [x] pypinyin 依赖 + requirements.txt
- [x] 文档同步: `doc/voice_control_design.md` 附录A
- [x] 29 测试用例全通过 (17 回归 + 3 字符模糊 + 9 拼音匹配)
- [x] 全局命令防误触: 单字 stop/exit 仅精确匹配 ("不停"不触发"停")
- [ ] 非预定动作序列: `_extract_gestures()` 精确+滑动窗口+拼音, 边界情况待调
- [ ] 多手势滑动窗口 refine: 长文本短词匹配 + 拼音误匹配

### 主流程优化 v15 (2026-07-17)
- [x] P3/HandInCup 超时双向延长: CRP 15s→20s + Python 10s→25s
- [x] 手动 CONTINUE 按键: 空格/'c' 最高权限覆盖
- [x] 人就位简化: only_person_detect → 纯人检测, 10帧稳定
- [x] Camera B 自动探测: camera_id=-1 → pygrabber 按名查找

### 语音控制基础设施 (v14, 2026-07-16)
- [x] ASR: sherpa-onnx SenseVoice int8 (227MB, 零PyTorch)
- [x] funasr 废弃: torch 2.12 输出乱码
- [x] `Omnihand/voice_omnihand_control.py` 完成
- [x] `doc/voice_control_design.md` 完整方案
- [x] requirements.txt 更新 (sherpa-onnx, sounddevice)

### v13 人就位 GPU 优化 + Cup P3 提速 (2026-07-16)
- [x] PersonDetector GPU化 + 合并双推理 + leave_frames 30→45
- [x] CRP sudu2 + 4处MovL提速

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 **+45 单独区分** | -30~+30 重拟合已完成 (28点, config 已同步); 相邻档误判用户已明确可接受; 唯一硬伤 = +45→15 跨档 (全局模型 ratio 档中心与 +15/+30 重合) → 需 u/v/d 判别面 |
| 🟡 多朝向示教 | 旋转检测就绪后，各朝向（±30°/±60°等）示教 → abc_offset 变化曲线 |
| 🟡 CRP 实机导入测试 | `estun_handle_sm.crp` + `estun_handle_cup_sm.crp` |
| 🟡 单相机 SM 实机联调 | `estun_handle_sm.py` + `estun_handle_cup_sm.py` |
| 🟡 豆包实机联调 (1/2/3) | 无手对话 → 空载手势 → 握手/急停/抢话/断网/音色 |
| 🟡 MiMo vs 豆包延迟对比 | 各 5 次，写回附录 C |
| 🟡 加水实测 | 完整流程 |
| 🟡 抓取姿态微调 | 末端 + 手势 |
| 🟢 RPS 手势集成到 Orchestrator | 4 阶段调度 |
| 🟢 序列+速度 实机测 | 连灵巧手测试 |
| 🟢 sudu2 实机验证 | CRP 导入示教器 |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯RGB无深度 (P3 X全示教)
- silero_vad.onnx 已通过 VoiceActivityDetector 接入 ✅ (之前 CFFI 回调崩溃已解决)
- ASR int8 量化精度不够 (已换全量 model.onnx 解决)
