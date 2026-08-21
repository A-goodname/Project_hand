# Progress — Project_hand

### Phase E RPS 接入: py + config + 一键脚本 (2026-08-21)
- [x] 用户写 CRP → 入库 `doc/estun_rps_phaseE.crp` (Down-667, sha a5054c91, 端口 8700; open→P1→While(独立 whileflag)→Home存储点→close; 位姿米/度)
- [x] config `estun_config.yaml rps:` 段 (port/done_flag/voice_log_path/arm_settle_delay/max_game_wait/rounds/poses.play/auto_invite)
- [x] 示教 P1 写入 `rps.poses.play` (RPSpos mm→m, 灵巧手落 RPS 相机 FOV)
- [x] `Estun/estun_rps_phaseE.py` 单一入口: X1→P1(米,度)→[RPS][RESULT]→DONE→等 EOF (Home 存储点, py 不回位姿)
- [x] `start_rps_phaseE.sh` 一键启动 (终端① RPS 服务 语音 stdout→日志, 本终端 phaseE py)
- [x] 新测试 6 例 (正常流程/no_rps/mock/全零拒绝/auto_invite FIFO/DONE 后 EOF) + 全量 148 测试绿
- [x] **while 只循环一次问题 (实机 2026-08-21)**: socketreadstr operateval 不能驱动 while 条件 (成功=0/超时=非0); Codroid 不能比字符串 → 用户改独立标志法 (whileflag) → Down-667; py 无需改 (DONE 内容不比)
- [x] **相机打开方式改 EyeToHand 式路径解析 (2026-08-21)**: `start_rps_phaseE.sh` 用 find_camera_id(usb-0:2+'UGREEN Camera') → RPS_CAMERA_INDEX, 不硬编码 /dev/videoN; ⚠️ find_camera_id stdout 清单打印需 `| tail -1`; RPS 相机 = Camera B = usb-0:2 = video6 (非 4.2)
- [x] **音响无播报修复 (2026-08-21)**: 双 PipeWire (wangxilu+user) 抢 JBL A2DP → user 挂不上 bluez sink; 停 wangxilu 音频 → 重连 JBL → bluez_output 出现 + 设默认
- [x] **语音不出声再排 (2026-08-21)**: realtime_audio `pw-cat --target bluez` 在 A2DP 静默失败 → 复制 config 清空 pipewire_target + `--config` 传语音 → 走 default sink (=JBL, paplay 实测有声); 不改 wangxilu 文件
- [x] **单局 demo 实机走通 (2026-08-21)**: 导入 `doc/estun_rps_phaseE.crp` + `start_rps_phaseE.sh` — 臂到位→auto-invite→JBL 播邀请→"开始"→判拳→结算→DONE→Home; 浏览器画面 + 声音正常
- [x] **多局制 rounds=0 等人退出 (2026-08-21, 代码完成待实机)**: config rounds 1→0 + quit_keyword; py 同时 tail [RPS][RESULT] 与裸"退出"行 (rps_voice_game 退出分支 print("退出"), 玩家说"退出游戏/不玩了"); rounds>=1 固定 N 局保留
- [x] **RPS 服务优雅收尾 (2026-08-21, 代码完成待实机)**: start_rps_phaseE.sh shutdown_rps() — phaseE 退出后 SIGTERM rps_stream+realtime_voice --rps-game (atexit 释放相机/串口), 兜底 fuser -k 8768, 免手动 Ctrl+C
- [x] 新增 5 测试 (退出触发/局后退出/精确匹配/超时兜底/rounds=2) + 全量 153 测试绿 (py_compile + bash -n 过)
- [ ] 实机复测多局制: 第 1 局→"开始"→第 2 局→"退出游戏"→DONE→Home→RPS 服务自动收尾
- [ ] 三 CRP 融合 (Phase0 8600 + A2D 8080 + PhaseE 8700, 段间 wait 1s)

### Phase 0 打招呼独立交付: py + CRP (2026-08-20)
- [x] 更新 doc/state_machine_reference.crp ← IDLEtemp-Down-645 (sha256 a8a8ee00, 状态机单独保存, 旧版 .bak)
- [x] data_sm-Down-946 与 doc/estun_handle_data_sm.crp sha256 相同 (4227c66b) — 已最新
- [x] 独立 Phase 0 CRP: 由示教器导出版取代 (socket IP 192.168.101.2→.96 修正), GH_* 存储坐标同步新示教值 → doc/estun_greeting_phase0.crp sha256 3c3f2bf5
- [x] 新建 Estun/estun_greeting_phase0.py: 独立 4 步 TCP, 人检测用迎宾 UGREEN usb-0:4.1, --mock 模拟
- [x] config greeting.camera_path: usb-0:4.1
- [x] 重新示教三个打招呼位姿 (2026-08-20): Z1/Z2/WS 更新入 estun_config.yaml + CRP GH_* 存储坐标同步
- [x] 排坑: socket 通信失败 = CRP socketcreate IP 旧 (192.168.101.2), 已改 .96; 通信失败时机器人回退 CRP 存储旧坐标
- [x] py_compile + mock TCP 4 步回复全对 + 142 测试绿
- [x] 实机验证 (2026-08-21): 示教器导入 estun_greeting_phase0.crp + PC 跑 estun_greeting_phase0.py 走通 4 步 (等人→挥手→握手→Home)
- [x] socket 时序核对 (2026-08-21): CRP SocketReadStr 10/130/30/90s vs py recv 5/130/30/90s — Step1/3 冗余足, Step2 余量~7s (人检测120s+YOLO加载), Step4 py 最坏 92.5s 略超 CRP 90s (~2.5s 缺口); 实机握手几秒完成, 用户判定冗余足够不修改; 备选: A=CRP 90→110~120s / B=py force_release_timeout 60→45s

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
  - [x] (2026-08-21 前置同步): CRP H_P3 后 wait 3s + H_P2/Home 改 MovJ; py 修 skip_frames 双计数 + PoseStabilizer θ 升级 + Phase B θ 插值/C 解析式/Z 钳位同步 data_sm (142 测试绿) — 待实机重测
  - [x] (2026-08-21 Camera A 生命周期 + 打印): Phase B 完成 → `_detect_active=False` 停检测; socket 断开 → True 恢复; HandInCup 耗时/18s 预警打印
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

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 **Phase D 失败根因待查** | CRP 到 C_P3 后 ~2s 自动停、READY 未发、连接关闭; 查 SocketSendStr/SocketReadStr 配置或时序 |
| 🔴 **data_sm 实机复测各档 P1** | θ 插值 + C 解析式 + Z 钳位链路落地验证; +15 档 Z 天然 364mm 观察钳位 |
| 🔴 **pos_side 实机只读验证** | 中间角度连续解算; 异常点 110950 (d=0.531 近距) 重采 |
| 🟡 实机验证 10 帧稳定窗口 | config 5→10 已定案; 训练侧申报证据合同 5→10 |
| 🟡 θ 稳定性记录工具实机逐档运行 | 分析 θ 逐帧 std → 再定稳定窗口/skip |
| 🟡 Phase 0 握手超时备选 | 若握手变慢: 方案A=CRP 90→110~120s / 方案B=py force_release_timeout 60→45s (现冗余足够) |
| 🟡 CRP 实机导入测试 | `estun_handle_sm.crp` + `estun_handle_cup_sm.crp` (遗留) |
| 🟡 单相机 SM 实机联调 | `estun_handle_sm.py` + `estun_handle_cup_sm.py` (遗留) |
| 🟡 豆包实机联调 (1/2/3) + MiMo 对比 | 无手对话 → 空载手势 → 握手/急停/抢话/断网/音色; 延迟对比 (遗留) |
| 🟢 sudu2 实机验证 | CRP 导入示教器 (遗留) |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯RGB无深度 (P3 X全示教)
- silero_vad.onnx 已通过 VoiceActivityDetector 接入 ✅ (之前 CFFI 回调崩溃已解决)
- ASR int8 量化精度不够 (已换全量 model.onnx 解决)
