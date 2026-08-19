# Active Context — Project_hand

> 工作区 /data/VLA/Project_hand(2026-08-16 起唯一;~/ 旧副本仅历史)。

## 当前焦点

**传统路线 Cup 阶段移植** — `estun_handle_cup_sm.py` + CRP 移植 Handle+Cup 全流程。Phase A-C（Handle 抓取→倒水→Cup 抓取→递送）已成功实机运行；**Phase D（READY→HandInCup→CONTINUE/ABORT）失败** — CRP 到 C_P3 后 ~2s 自动停止运动，未发送 READY 到 Python，连接关闭。根因待查（CRP SocketSendStr/SocketReadStr 配置或时序问题）。

**多视角朝向示教（本地主线）** — 正对场景完整流程已通过实机验证（data_sm + handle_teach + CRP，2026-08-17，抓壶→倒水→放回→Home 全过）；handle_teach.enabled: true 实机低速验证完成。**方案 A（宽高比法）旋转检测已集成进 handle_cup_detector.py**（2026-08-18）：判档规则 = 用户定的 6 档固定抓取 `+45/+30/+15/0/-15/-30`（无 -45），`[-10,10]=0, (10,20]=15, [20,40]=30, >40→45, <-40 不可达`。实现：正半侧 (θ>10) 用 **per-band 位置模型 r̂_a(u,d) 最近邻**（45/30/15 三档，规避 30° 端 θ 反解放大 + 45° 反解失效），负半侧用反解 θ 区间（负侧无 per-band 数据，反解已证可靠）；**45 档需 u≥530 门限**（u<530 与 0° 物理不可分）。模型系数/门限写入 `Orchestrator/config.yaml handle_rotation` 段。**判档正确性**：标定 29 点离线验证 28/29 判对、29/29 可接受（pos45→30 为用户明确可接受，无 45→15）。**抓取姿态规律**：绕竖直轴旋转只改 C（`C=C0-θ`，误差<0.2°），A/B 恒定。**band 实机验证结论（2026-08-19）**：0° ✅ 全对；+15° ⚠️ 94% 判 15（6% 边界抖落 0）；**-15° ❌ 判 0 为主**（反解低估 ~5° + 无 -15 标定数据）；**+45° ❌ 判 15**（45 per-band 模型过拟合 + 遮挡致 ratio 与 +15 档物理重合 0.294 vs 0.294）。**用户定方向**：补 -15 数据 → 拟合 **-30~+30 主模型** → **+45 与易混淆档用 u/v/d 等参数单独区分**。T-Offset 定案维持：姿态 ABC 用 C=C0-θ 规律不示教，每档 t_offset 需重新示教。**旋转判档已简化定版 (2026-08-19)**：纯 θ 区间 5 档（[-8,8]=0, (8,22]=15, >22=30），无 ±45，+45 统一落 +30。下一步: 逐档示教 t_offset（teach_p1 需支持角度参数）→ C 补偿 + band 查表 t_offset 接入 data_sm P1。

训练侧线（**已完成**）：P2/P1 双确认协议适配（D0 门禁）协作线已交付完毕。

## 最近变更

### t_offset 5 档示教完成 (2026-08-19)

1. **teach_P1 --angle 逐档示教完成**: p1_offsets 表 5 档 (-30/-15/0/+15/+30) 全齐
   - 0°: [33.0,-83.4,2.2]mm / C=-21.1 (2026-08-17 默认值)
   - +15°: [10.8,-90.0,2.8] / C=-36.3; +30°: [-17.9,-87.3,-0.5] / C=-51.1
   - -15°: [67.0,-68.3,-0.8] / C=-6.2 (Z 方差 6.2mm 略超验收, C 预测-6.1 实测-6.2 吻合)
   - -30°: [82.9,-47.6,-5.0] / C=+9.0
2. **C=C0-θ 全档验证 (误差 ≤0.2°)**: ΔC vs 0° = [-30:+30.1, -15:+14.9, +15:-15.2, +30:-30.0] — 完美线性
3. **X 偏移单调**: -30:+83 → -15:+67 → 0:+33 → +15:+11 → +30:-18 mm (≈-1.7mm/°, 负侧略弯)
4. **-15 预测验证**: 预测 [X58/Y-65/C-6.1] vs 实测 [X67/Y-68/C-6.2] — C 完美, Y 接近, X 偏 9mm (X-θ 负侧非线性)
5. **注意**: -15 的 abc_offset C 方差 169° 是 ±180 环绕假象 (p1_abc C 干净); -15 Z 方差 6.2mm>5mm (量级小可接受)
6. **待办**: ①data_sm 稳定逻辑 (θ 5帧中位数→band, 用户要求示教完成后做) ②data_sm 查 p1_offsets 表 + band 接入 (task #4)

### teach_P1 支持角度参数 (2026-08-19)

1. **`estun_teach_handle_p1.py` 新增 `--angle`**: 示教时指定档位角度 (deg, 如 `--angle 15`), 打角度标签
2. **输出**: banner/log 头部记录示教角度; yaml 输出块改为按角度建 `handle_teach.p1_offsets.<angle>` 表结构
3. **`--apply`**: 写入 `handle_teach.p1_offsets.<angle>` (按档 t_offset 表, data_sm 用) + 平铺字段同步本档 (旧 data_sm 兼容)
4. **用途**: 逐档示教 t_offset (task #3) — 5 档各跑一次 `teach_P1 --angle <band>`; 完成后 data_sm 按检测 band 查 `p1_offsets` 表

### 5 档简化定案: 纯 θ 区间, 无 ±45 (2026-08-19)

1. **用户定案 (简化)**: 旋转判档只保留 **5 档 -30/-15/0/+15/+30** (无 ±45); +45 物理摆位统一落 +30 (此前已接受); 每档分别示教 t_offset
2. **判档边界 (用户 2026-08-19)**: **[-8,8]=0, (8,22]=15, >22=30**; 负侧对称 [-22,-8)=-15, <-22→-30, <-40 不可达 — **纯 θ 反解区间**, 删掉 per-band 位置模型 / u 门限 / 45 识别
3. **代码**: `_classify_rotation` 重写为纯 θ 区间 (5 分支); `__init__` 只加载 band0_halfwidth=8 + band15_hi=22; 删除 _rot_band/_rot_u45/_rot_u_high
4. **config**: handle_rotation 删 band_coeffs/angle45_gate/u_high_zone; 保留 global_model (28点) + band0_halfwidth + band15_hi
5. **全量验证 (35 点)**: 精确 26 / 相邻 7 / 跨档 2 (pos45/pos45_3 → 15, 低 u≈533 非操作摆位); -30/-15/0/+15 全对, +30 3/5→30 + 2/5→15 (相邻)
6. **实机含义**: 干净 +30 (u≈513, θ~16) → **15 band**; +45 (u>560, ratio 0.34+, θ=29.6) → **30 band**; 5 档对应物理 ≈ -30/-15/0/「15」=物理+15~+30/「30」=物理+45
7. **JSON**: model 段更新为 θ 区间描述 (删过时键); +45 历史点保留备查
8. **待办**: 逐档示教 t_offset (5 band, teach_p1 需支持角度参数) → C=C0-θ + band 查表 t_offset 接入 data_sm P1

### +45 真相: 裁切是高 ratio 的来源, ROI 扩展回退 (2026-08-19)

1. **u>560 门限定案**: config `angle45_gate.u_min` 530→560 (低 ratio 簇 u≤540 与 +15 物理重合, 不再认 45)
2. **高 ratio 簇确认**: pos45_1/2/100600 (u 587~613, ratio 0.34~0.36) 判 45 ✓
3. **真相发现**: 尝试 ROI 上下扩 20% (`TAG_ROI_VPAD=0.2`) 防裁切 → 同一位置 ratio 从 0.3598 掉到 0.2947~0.2995, **长轴恢复** (49.5→59.1) — **0.34~0.36 的"高 ratio"就是贴纸被 YOLO 框裁掉的产物**; 干净 +45 ratio ≈0.295~0.30 落在 +15 区间, 纯 ratio 根本分不开
4. **用户决策 (回退)**: 贴纸被裁是 +45 的**稳定特征**, 回归原 bbox 高度检索 (`TAG_ROI_VPAD` 已删, handle_cup_detector + test_tag_points 同步), 保留裁切效应 → ratio 虚高 0.34~0.36 才能与 +15 (0.30) 区分; 校准数据本就是旧管道采集, 回退后操作与校准一致
5. **扩展代码采集的两个干净点 (pos45_101756 u602/r0.2947, pos45_102144 u599/r0.2995) 不作数** — 管道与操作不一致, 未入档
6. **+45 定案 (u>586 高角度区, 2026-08-19)**: `_classify_rotation` + config 新增 `u_high_zone: 586` — **u>586 时候选只留 {30,45}** (per-band 比 r̂30/r̂45); 依据: 校准中 +45 独有 u≥587, +15/+30/+0 最高 u=586。补入裁切点 pos45_101155 (0.3598)/pos45_102457 (0.3144); pos45_093515 (u540) 标 excluded (+45 域外, 反推 θ<10 判 0)
7. **全量复验 (35 有效点)**: +45 7/7 达标 (4 判 45 / 3 判 30, 无 45→15/0); 其余 30 精确 + 3 相邻 (neg15_094913→0, pos15_3→0 可接受) — **真实跨档 0**
8. **注意 (脆弱点)**: u_high=586 与 0°/30° 最高 u (585.8/586) 相邻 — 实机 0°/30° 摆位必须保持 u≤586, 否则被误路由到 30/45 区
9. **待办**: 实机验证 u_high 门限 (各朝向 band 输出) → 逐档示教 t_offset → C 补偿接入

### -15 补数据 + 28点重拟合 + 相邻档可接受标准 (2026-08-19)

1. **-15° 三点入档** (用户提供 3 个实机 log 提炼, std≈0.001): neg15_094705 (ratio0.2567/u541.6/d0.653) + neg15_094913 (0.2650/u552.6/d0.627) + neg15_095007 (0.2536/u525.6/d0.682); JSON data_points 36 点
2. **重拟合 -30~+30 (28点: -30×5/-15×3/0×11/15×4/30×5)** — `cos+sin+dd`: c0=0.1071, c1=0.1739, c2=0.0988, c3=-0.0768, d_ref=0.63; R²=0.9780 (含 -15 后略降, 因 -15 物理点非纯正弦), LOOCV ratio MAE=0.0046; φ=29.61°, A=0.2000
3. **反解角度偏差 (每档)**: -30 [+0.3,-2.0] / **-15 [+3.4,+6.6] 系统性低估~5°** / 0 [-4.8,+2.1] / +15 [-5.3,+1.6] / **+30 [-11.1,-0.4] 正侧饱和** — 与实机 -15→0、+45→15 现象同源 (ratio 响应在正侧/负15 端被压缩)
4. **三种判档方案对比 (28点)**: 方案A 现状代码 (正半侧per-band+负半侧反解区间) 26/28; 方案B 档中心最近邻 (全局模型在每档角度预测ratio, argmin|实测-预测|) 27/28; 方案C 全档per-band u,d 27/28
5. **用户新标准 (2026-08-19)**: **相邻档误判可接受** (如 -15→0/-30 均可), 只需无跨档 → 三方案全达标 (误判全为 ±1 档); **之前标记的 -15 实机 76%→0、pos15_3→0 均撤销为可接受**
6. **config.yaml 同步**: handle_rotation.global_model 更新为新系数; 注释标注 28点重拟合 + 相邻档可接受标准; JSON model 段已更新 (bands/band_centers/classification=档中心最近邻建议)
7. **唯一硬伤仍是 +45→15 (跨两档)**: 全局模型预测 45° 档 ratio≈0.300 与 +15(0.301)/+30(0.307) 几乎重合, 纯 ratio 不可分 → 需 u/v/d 单独区分 (task #7)
8. **推荐判档方案**: 方案B 最简单且无 per-band 过拟合 (C 的 -15 档 3 点 3 参数是退化拟合), 正负两侧统一; 是否把代码从方案A改B待用户确认
9. **待办**: +45 单独区分 → 逐档示教 t_offset → C 补偿接入

### band 实机验证 + 方向调整: -30~+30 主模型 + +45 单独区分 (2026-08-19)

1. **实机验证四档** (handle_cup_detector 独立跑, 只读相机):
   - **0° ✅**: band=0 全对, ratio≈0.279 (落标定 0° 档中心), th 波动 0~7.8 但稳判 0
   - **+15° ⚠️**: 稳定段 34 帧 32 判 15 (94%), 2 帧 ratio 掉 0.2897/0.2906 → th 9.0/9.5 → 判 0。0° 最大 th=7.8 vs 15° 最小 9.0, 缝隙仅 ~1.2°
   - **-15° ❌**: 42 帧仅 24% 判 -15, 76% 判 0。th 只有 -8.4~-10.6 (期望 -15), **反解系统性低估 ~5°** + **-15° 档 JSON 零数据** (纯反解区间映射)
   - **+45° ❌**: 实机 log (u≈540, d≈0.626) 稳定 ratio≈0.2944。per-band 最近邻**判 15** (r̂45=0.311 过拟合偏高, r̂15=0.294 与实测重合), **无一帧判 45**。违反硬约束 45→15
2. **+45 物理混淆根因**: 45° 遮挡致 fitEllipse 短轴未按理论 cos45° 收缩 (理论 ratio≈0.19, 实测 0.294), 与 +15° 档实测 (0.293) **几乎重合**。标定数据 +15 [0.2947,0.3038] 与 +45 [0.299,0.3556] 本就重叠
3. **实现与设计不符**: 标定阶段 45 门限 = `ratio≥0.2925 且 u≥530`, 但 `_classify_rotation` 实现成 **per-band 最近邻** (u≥530 仅把 45 加入候选, 无 ratio 门限); JSON band_coeffs 无 45/-15, config band_coeffs 有 45 (4 点拟合, 过拟合, b1=0.000764 是其它档 5~60 倍)
4. **用户定方向 (2026-08-19)**: ①补 -15° 数据 ②拟合 **-30~+30 主模型** (不含 +45) ③**+45 与易混淆档用 u/v/d 等参数单独区分**
5. **JSON 已补 +45 实机点** (pos45_093515_实机: u539.9/d0.626/ratio0.2944/std0.0031/n50), 现 33 点; 验证重拟合 45 (5 点 R²=0.85) 后实机点仍判 15 → 证实补数据不能解决, 需换区分方案。config.yaml 未动 (恢复备份)
6. **待办**: 等用户补 -15 → 重拟合 -30~+30 → 设计 +45 单独区分 → 逐档示教 t_offset → C 补偿接入

### T-Offset 方案定案: 每档示教 + 姿态规律 C=C0-θ (2026-08-19)

1. **方案定案 (用户)**: 每档抓取**姿态 ABC 不用示教** — 参考 C 变化规律 (`C=C0-θ`, A/B 恒定) 自动补偿; 但**新的档位需重新示教 t_offset** (位置偏移方向随把手旋转, 固定 p1_offset 不成立)
2. **实机验证结果**: -30 档可行; **+45→+30 可容许** (45 判成 30 是用户明确可接受的, 与离线判档结论一致)
3. **待办**: 逐档示教 t_offset → data_sm P1 = 检测位置 + 该档 t_offset (姿态 C 按 C=C0-θ 补偿)

### T-Offset 需随 band 调整 (2026-08-18)

1. **用户指出**: band 档位下**位置偏移也要随旋转调整**, 不能只用固定 `p1_offset` — 固定偏移 (0° 示教值) 在旋转档位下抓取点不再对准, 位置补偿不只是 detect_base 自动跟随 + C 补偿
2. **影响**: 之前的推论 "旋转后位置检测自动跟随, P1 = detect_base + p1_offset 仍适用" (见下方 抓取姿态规律 #2) **被推翻/需修正** — 把手旋转后, 检测点→抓取点的偏移方向随把手旋转, 固定基系偏移不成立
3. **方案方向**: 每档位置偏移需按 band 调整 (具体机制待定: 可能按旋转角旋转偏移向量 / 每档补示教); **待实机示教/标定确认**
4. **待办**: 修正 data_sm 的 P1 计算 (位置偏移随 band 变 + C 补偿), 需实机验证各档落点

### 旋转检测集成进检测器 (2026-08-18)

1. **用户定档位规则**: 6 档固定抓取 `+45/+30/+15/0/-15/-30` (无 -45); `[-10,10]=0, (10,20]=15, [20,40]=30, >40→45, <-40 不可达`; 硬约束 **45→30 可接受, 45→15 不可接受**
2. **实现策略 (正负分离)**: 正半侧 (θ>10) 用 per-band 位置模型 r̂_a(u,d) 最近邻判 45/30/15 (45 需 u≥530 门限) — 规避 30° 端 dr/dθ→0 反解放大 (pos30_4 曾错 -10°) + 45° 档全局反解失效 (pos45_1/2 ratio 超 A 被 clip 到 32°); 负半侧用反解 θ 区间 (负侧无 per-band 数据, 反解 -30 档 5/5 可靠)
3. **判定验证 (标定 29 点)**: 28/29 判对 + 29/29 可接受; 唯一非判对 = pos45→30 (用户明确可接受); **无 45→15/45→0**; 30° 档 5/5, 15° 档 4/4, 0° 档 11/11, -30° 档 5/5
4. **接入文件**:
   - `Orchestrator/config.yaml`: 新增 `handle_rotation` 段 (enabled + global_model 系数 + band_coeffs 45/30/15/0 + angle45_gate.u_min=530)
   - `Orchestrator/handle_cup_detector.py`: `__init__` 加载 handle_rotation; 新方法 `_solve_theta` (θ=φ-arccos(...), 负分支, clip ±1) + `_classify_rotation` (返回 (band, theta_deg)); detect() 在 HSV 成功时算 ratio→θ→band, handle_result 新增 ratio/theta_deg/band 键; visualize + run() 终端输出 band; 失败分支重置
5. **离线验证**: py_compile + 几何测试 4/4 + rot_math 15/15 + state_machine_dynamic 8/8 + site_gate 5/5 全过
6. **待办**: 实机跑检测器验证 band 显示 → C 补偿 (C=C0-θ) 接入 estun_handle_data_sm.py 抓取姿态

### JSON 更新 + 25 点重拟合 + 0/45 单独区分 (2026-08-18)

1. **JSON 入档 9 新点 → 32 点**: +30×2 (pos30_4/5) + +15×3 (pos15_1/2/3) + 0°×2 (0_new_1/2) + +45×2 (45_u491/45_u494); 45_u490 两点标 excluded (门限外)
2. **+45 档 4 点恢复有效** (pos45/pos45_1/2/3, u≥530): 用户确认 0 与 +45 单独区分成立后, excluded 改 false; **u<530 的 45 保持 excluded** (与 0° 物理不可分)
3. **重拟合 -30~30 (25 点, 剔 pos30_3)**: 对比 6 种特征组合 (cos+sin / +dd / +u+dd / +ud / +du / full) — `cos+sin+dd` 已最优 (LOOCV 0.0036, 角MAE 2.60°); 加 u 仅 LOOCV 0.0036→0.0035 但角MAE略升 2.64°, 无实质提升; full 过拟合 (LOOCV 0.0040)
4. **前后对比 (cos+sin+dd)**: 旧18点 R²=0.9925/LOOCV 0.0037/角MAE 2.34° → 新25点 R²=0.9911/LOOCV 0.0036/角MAE 2.60° — **质量持平** (LOOCV 略好; 角MAE 略升因新增 +15 档本就弱, 非模型退化)
5. **含 v 组合**: 仅 17 点有 v (+15 档 4 点全缺, 早期点也缺), 17 点子集 u+v+dd vs u+dd 对比 R² 增量 0.0000 — **v 无贡献, 弃用 v**
6. **per-band 位置拟合 LOOCV 分类 88% (22/25)**: -30: 5/5, 0: 10/11 (1→15), 15: 2/4 (1→0, 1→30), 30: 5/5 — **0↔30 零混淆** (硬需求满足), 错误全在 15° 档相邻混淆 (15° 档带宽 0.013 < 间隙, 已知弱点)
7. **0 vs +45 单独判别**: 0° 档 ratio [0.2727, 0.2861] (11点) vs 45° (u≥530) ratio [0.2990, 0.3556] (4点) → margin +0.0129; 门限 `ratio≥0.2925` 且 `u≥530` 判 45: 4/4 + 11/11 全对; u490 两 45 点在门限外 (0.2748/0.2835 落入 0° 档)
8. **新模型已写回 JSON**: coefficients (c0=0.1147, c1=0.1642, c2=0.1027, c3=-0.0725, d_ref=0.63), φ=32.02°, A=0.1936, R²=0.9911/LOOCV 0.0036/角MAE 2.60°; + `band_coeffs` (各档 r̂_a(u,d) 系数) + `compare_before_after` + `angle45_gate` 段
9. **待办**: per-band 判档系数已写 JSON, 集成检测器时直接用

### Cup 阶段移植 + Camera B + Phase D (2026-08-18)

1. **estun_teach_cup.py 新建**: Cup P1 抓杯点示教工具 (镜像 estun_teach_handle_p1.py), 检测水杯基系位置 + 人工示教 → t_offset (基系直接加); 3 次示教验收: t_offset (-0.0226, 0.0021, -0.0078) m, 方差 (3.5/4.2/3.1)mm ✓
2. **estun_config.yaml 新增 cup_teach 段**: enabled: true, p1_offset + p1_abc + p3_pose (示教固定值 [-553.2, -11.1, 479.8, -91.9, 88.9, 66.0] mm/deg)
3. **estun_handle_cup_sm.py 修改**:
   - handle_teach + cup_teach 双链 (与 data_sm 对齐)
   - Camera B (usb-0:2, 传统路线) + PersonDetector (detect_person_only) + HandInCupDetector (update + is_hand_on_cup)
   - Phase D READY/RELEASE/CONTINUE/ABORT 逻辑 (镜像 state_machine.py)
   - camera.py path+name 联合匹配修复 (防 D435i 同控制器误匹配)
4. **EyeToHand/config.yaml**: camera_path 改 usb-0:2, cup_roi_fallback 重标 (327,133)→(394,210)
5. **CRP Phase D 添加**: 从参考 CRP 克隆 blocks (TranStrToCpos + If 分支), 新增 points (DummyCpos, C_P4, DCp2, C_P5, DCp3), sudu2 SPEED 变量; 用户手动调整 READY/RELEASE 发送 + Wait 时序
6. **Phase A-C 实机验证通过**: Handle 抓取→倒水→放回→Cup 抓取→递送 (C_P3) 全流程正常
7. **Phase D 失败**: CRP 到 C_P3 后 ~2s 自动停止运动 (无报错), 未发送 READY 到 Python, 连接关闭。If 分支未进入 (strflag_cont/abort 保持初始值 1)。根因待查: CRP SocketSendStr READY 是否实际发送 / SocketReadStr 时序 / TranStrToCpos 解析
8. **P3 X+1cm 微调**: handle_grasp.p3_teach_offset.left.30.t_offset X: 0.0686 → 0.0786 m
9. **doc 清理**: 删除旧 session-barrier 文件 (7 个), 更新 data_sm/cup_sm CRP 为最新版

### 抓取姿态规律 + 位置影响分析 (2026-08-18)

1. **姿态随旋转规律 (用户示教数据验证)**: 绕竖直轴旋转 θ 只改 C, A/B 恒定 — 0° C=-21.1 → -15° C=-6.276 (预测差 0.176°) → -30° C=+8.919 (差 0.019°); A 偏差 ≤0.37° / B ≤0.04°; 规律 `C = C0 - θ` (θ 顺时针为正, 与 ratio 模型一致)
2. **推论**: 姿态补偿只需转 C, **无需每档预示教姿态**; 位置检测自动跟随 (旋转后 detect_base 变); ~~P1 = detect_base + p1_offset 仍适用~~ — **已被 2026-08-18 用户纠正: T-Offset 需随 band 调整, 固定 p1_offset 不成立** (见上方 "T-Offset 需随 band 调整"); 记入 JSON `grasp_pose_calib` 段
3. **位置(uv)影响分析 (判档方案转向)**: r 随 u 正相关 / 随 d 负相关 (透视+畸变); 0°档 9 点宽 0.0134 / 30°档宽 0.0137, 而 0↔15 间隙 0.0086 / 15↔30 间隙 0.0093 — **档内展宽 > 档间间隙, 纯 r 阈值必然错分**
4. **判档方案改为档内位置拟合**: 每档 a∈{0,±15,±30,±45} 拟合 `r̂_a(u,d) = b0+b1·u+b2·d`, 运行时 `argmin_a |r_实测 − r̂_a(u,d)|`, 距离超阈值 → "无法识别"; 比全局 θ 反解更适固定姿态方案 (规避 30° 端 θ 反解放大)
5. **待补示教数据**: +15(1→3-5位置) / +30(3→+2-3) / +45(重采3-5, 位置关系显著) / -15(0→3-5) / -30(可选); 0° 已有 9 位置够用

### 方案 A 模型定版: [-30°,+30°] 工作范围 (2026-08-18)

1. **数据**: `logs/ratio_angle_calibration.json` 23 点 → 45° 四点弃用 (遮挡/内部自矛盾 ratio 0.299~0.356, 含 45° 时 R² 从 0.988 掉到 0.92) → 19 点 (-30/0/15/30) → 剔 pos30_3 离群 (std=0.0054 最大, ratio 0.294 vs 同组 0.311/0.318, 反解 12° 非 30°) → **18 点定版**
2. **定版模型 (cos+sin+dd)**: `ratio = 0.1202 + 0.1574·cosθ + 0.1029·sinθ - 0.0835·(d-0.6399)`; R²=0.9925, LOOCV MAE=0.0037, 角度反解 MAE=2.06° / max=9.7° (+30° 端 dr/dθ 最低, 灵敏度天然弱)
3. **方向不对称根因确认**: 水壶面向桌子长边/XZ 平面非面向相机 → 贴纸相对相机初始偏角 α≈28° → 顺时针 vs 逆时针响应不对称 (r_cw/r_ccw≈1.5); sinθ 项捕捉; 纯 cosθ 拟合差 (R²=0.02), cos+sin 达 0.98
4. **位置依赖**: r 同角度随 u (水平像素)/d (深度) 变化 (镜头畸变+透视), 故加 dd 项; 物理偏角 α 模型弃用 (用户判定直接数据拟合更可靠)
5. **反解公式**: θ = φ - arccos((ratio-c0-c3·(d-d_ref))/A), φ=atan2(c2,c1)=33.17°, A=√(c1²+c2²)=0.1881; **|θ|>30 归档 ±45**
6. **待集成**: 反解逻辑 + 档位判定 (|θ|≤30 连续角度, 超限 ±45) 接入 `handle_cup_detector.py`

### 旋转敏感性验证 + 宽高比法方案 A (2026-08-17)

1. **P1 示教 (HSV 链, 正对 4 位置) 验收全过**: t_offset 方差 2.0/1.7/2.0mm, abc_offset 0.2/2.5/2.0°; euler_cam A/B 恒定 (180,0) 仅 C 随位置变 (几何效应); 相对旧 SAM 链 A 漂 32°→0.2°; 已发训练侧 `logs/teach-p1-hsv-analysis-2026-08-17.md` + log (sha256 校验过)
2. **旋转实验 (关键发现)**: 水壶绕竖直轴 ±30°, fitEllipse 长短轴姿态 euler C 仅反映 ~3° — **2D 图像方向对绕竖直轴旋转不敏感** (旋转轴 ∥ 图像竖直, 数学必然); 单纯图像姿态无法区分朝向
3. **3D 点云 PCA 尝试 → 回退**: 旋转敏感但噪声大 (法线 λ3 由深度噪声主导, B 分量波动 6-14°); 方向语义 (grasp_direction: Z 朝相机/Y 指向图像下上边界) 随回退未保留 — **后续重做姿态时需补回**
4. **方案 A (宽高比法, 进行中)**: r = 短轴/长轴 = r0·cosθ (贴纸标准矩形, 旋转投影宽度收缩), θ = arccos(r/r0); 已实现于 `Orchestrator/test_tag_points.py` (ratio/theta 输出 + `c` 键标定 r0), 检测器未动; **r0 = 0.274 已标定 (±0.001, θ 精度预期 0.2-0.5°)**
5. **待验证**: ratio 平移不变 (不同位置正对) + ±30/±60° 旋转响应 + 顺逆对称; ±歧义未解决 (转动历史 / bbox 侧向偏移消歧备选)

### P3 修正 + teach_p3 换源 (2026-08-17)

1. **实机问题 (data_sm 完整流程首次实跑)**: ①P1/P2 落点正常 (handle_teach 链 ✓) ②**P4/DCp1** 为运行时 CposOffset 计算 (H_P3−H_P1 差值写入 DCPOS, 初始 0 正常, 无需改) ③**P3 逆解失败** — 姿态 AC 跳变: p3_teach_r_abc (-90, 30.3, -90) 是旧固定 ABC 时代值, 与 handle_teach 后 P1 (-115.2, 85, -21.1) 不匹配 ④**P3 位置不对** — teach_p3 硬编码 Home (387,100,616)/(-118.6,0.8,-91.8) 已过时, 当前 Home = [-111.901, 378.274, 698.33, -125.647, 0.759, 4.803] (mm/deg, 与示教 R0 附近一致), Cup 基系转换全错
2. **data_sm P3 姿态**: 完整固定 ABC = p3_teach_r_abc (示教值, **与 P1/P2 姿态不要求一致** — 用户约定 2026-08-17; 曾试 AC 继承 P1 合成, 用户否决回退)。逆解失败根因 = 旧 Home 致 P3 位置不可达 + 旧 r_abc 过时, 重标后直接用示教值
3. **estun_config.yaml**: 新增 `home_pose` 段 (mm/deg, 与示教器 CRP Home 同步, 2026-08-17)
4. **estun_teach_p3.py 换源**: 检测器 RealSense.test_cup_6D (旧链) → HandleCupDetector (Orchestrator, 与 data_sm 同源); Home 从 config 读 (不再硬编码); 稳定窗口只用位置 (cup 无动态 euler); compute_cup_base 删 cam_euler 死代码; **输出格式未动**
5. **t_offset 判定**: P1/P2 实机 OK + 方差验收过 → 但用户决定重新标定 (t_offset 重标进行中)
6. **P3 重标完成 (teach_p3, 2026-08-17)**: 新 Home + HandleCupDetector 同源; 3 次示教 (第 2 次 cup 深度 z 异常剔除, 第 3 次方向键转义丢记录已补记); 汇总 t_offset (0.0686, -0.0094, 0.2536) + r_abc (-93.3, 30.5, -5.3) → 已写 config p3_teach_offset.left.30
7. **排坑 (输入解析)**: 终端方向键误按 → ^[[D 转义混入 input() → float 解析失败丢示教; teach_p3/teach_p1 均已加 ANSI 转义剥离
8. **完整流程实机验证通过 (2026-08-17 17:19)**: data_sm + 新版 CRP 完美运行 — P1/P2 (handle_teach 链) / P4 (DCp1 动态回退) / P3 (重标位置+固定 ABC) 全部正常; t_offset 用户判定不用重标 (handle_teach 保持 8-17 值)
9. **CRP 更新 (2026-08-17)**: 用户修好诸多问题后回传 `data_sm-Down-638.crp` → 已替换 `doc/estun_handle_data_sm.crp` (保持命名; 旧版备份 `.bak`, sha256 旧 0a4aa33→新 1e88b99)。新版 38 widgets (旧 35): equal 5→7、movl 7→8 — **新增 P3 倒水后 `MovL H_P1` 放回水壶步骤**; 流程 = P2→P1→PREGRIP/GRIPDONE→P4(CposOffset DCp1+H_P1)→P3(sudu)→P4→H_P1(放回)→RESET→P2→Home; TranStrToCpos lengthunit=m ✓ / CposOffset pointa/pointb ✓; points 新增 P1 (9 个)
10. **提交 + 回传 (2026-08-17)**: 提交 08b067a (7 文件 +191/-75, 本地不 push); 回传训练侧 inbox (sha256 对方校验全过): `estun_handle_data_sm.crp` (1e88b99) + `git-diff-144c025..HEAD.patch` (b0b3f22) + `project-hand-pour-flow-frontal-2026-08-17.md` (eba3554, 说明"正对时倒水流程") + sha256.txt; staging 目录 ~/p2p1-handoff-staging/

### data_sm 接入 handle_teach: 示教偏置进运行时完整流程 (2026-08-17)

1. **背景**: 正对示教验收通过 (t_offset 方差 2.0/1.7/2.0mm, abc_offset 0.2/2.5/2.0°) 后, 把偏置接入运行时。用户定案: 完整流程 = 抓壶→倒水→放回→Home, 无 Cup 阶段, cup 检测定 P3 倒水点, 姿态仍固定 ABC (参考 log p1_abc)。**落点 = `estun_handle_data_sm.py`** (该状态机已具备全部骨架: cup 稳定器 + P3 倒水点 + 无 Cup 阶段; CRP `doc/estun_handle_data_sm.crp` 35 widgets 现成: P2→P1→DCp1→P3→DCp1→P1→RESET→P2→Home)
2. **estun_config.yaml**: 新增 `handle_teach` 段 (参考 8-17 log: `p1_offset [0.0330,-0.0834,0.0022]` / `abc_offset [-169.2,76.2,-19.7]` 备用 / `p1_abc [-115.2,85.0,-21.1]`), `enabled: false` 先对照, P1 落点验证通过后置 true
3. **estun_handle_data_sm.py 三处**: ① `__init__` 补属性 ② `_load_configs` 加载 handle_teach, enabled 且含 p1_abc 时覆盖 `_handle_fixed_abc` ③ Phase B P1/P2 位置**双链切换** (enabled → P1 = detect_base + t_offset 基系直接加, **跳过 t_handle_effective**; P2 = P1 + R_base_fix @ pre_offset; 否则旧链 ee 系减 t_eff 对照)。P3 倒水点/Phase C/CRP 未动
4. **关键约束 (排坑)**: handle_teach 链 (基系直接加, 含 TCP 语义) 与 t_handle_effective ee 系补偿链**数学语义不同, 不能混用** — enabled 二选一; P2 推导式 `P2 = P1 + R_base_fix @ pre` 由 `R_ee_base@R_grasp_ee == R_base_fix` 恒等保证 (数值验证 True)
5. **离线验证全过**: 示教 log 4 次回放 (cam_mean → 新链 P1) 误差 0.7-3.2mm (在示教方差内, 新链复现示教落点); P2 双链各自内部一致; 新旧链 P1 差 11.4mm (语义差, 对照意义)
6. **待实机**: 只读跑 data_sm (不连机器人) 看 [Handle STABLE]/[Cup STABLE] → Phase B 发送前 [TEACH] 打印与示教 P1 对比 → enabled: true 完整流程低速验证

### 检测器 HSV 化: SAM 链 → 青色贴纸链 (2026-08-17)

1. **背景**: SAM mask 不稳定 (覆盖区域不一致/偶发失败 → 中心/姿态漂移); 把手正面贴青色贴纸 (标准长方形), HSV 分割替代 SAM
2. **新链 (handle_cup_detector.py)**: YOLO kettle bbox → **bbox 右 30%** HSV 青色分割 (H 40-110 / S≥50 / V≥50) → 最大域 → 质心=中心 (3D) + **长短轴=6D 姿态** (X=短轴, Y=长轴, Z=X×Y, z<0 朝相机) → 帧间锚定 + 矩阵 EMA → euler_cam
3. **深度排坑 (3 个)**: ① **color/depth FOV 不同 → 必须 color_to_depth_pixel 映射** (直接取彩色坐标错位到背景, z=1.6m 根因) ② 深色贴纸吸红外 → 质心邻域中位数 + 区域内中位数兜底, **不取区域外** (贴纸盖满把手面时外扩环形带=背景) ③ fitEllipse 的 angle 对应 width 边 → 长短轴按 axis1/axis2 大小取 (90° 错位坑)
4. **HSV 参数实机标定**: 贴纸为青绿渐变 (H 36-105, S 52-104), `HUE_LOW` 70→40; 白色 (S 低) / 黑色盖子 (V 低) 被排除
5. **实机验证**: 中心 ±0.2px / euler 0.1-0.2° / 深度 0.63m ✓ (test_tag_points.py 轻量链: D435i + YOLO, 无 SAM)
6. **SAM 链**: 移除模型加载 (_sam_model 恒 None), 方法定义保留未调用; 单测 31 例全过
7. **试验脚本**: `Orchestrator/test_tag_points.py` (轻量 HSV 验证链, 含拖拽圈定 HSV 统计) + `mark_roi_hsv.py` (静态帧圈定分析)

### P1 示教工具实机适配: 非连续帧 + R0 延后 + 窗口开关 + 欧拉约定验证 (2026-08-16)

1. **实机问题**: SAM 成功率有限 (约 1/4-1/5), 原稳定窗口要求**连续** 3 成功帧, 任一失败帧 `buf=[]` 清空 → 窗口永远凑不齐, 示教无法进入
2. **修复 (estun_teach_handle_p1.py)**:
   - 失败帧不再清空 buf — 示教场景机械臂静止+水壶不动, 成功帧累积即可; 新增 `BUF_MAX_AGE=5s` (相邻成功帧间隔超时重置, 防位置漂移)
   - 姿态波动门限: 窗口内 euler 逐分量 ±360k 对齐 (防 179°/-179° 环绕误判) 后极差 <5° (`ANGLE_THRESH`) — **位置+姿态双门限**才判稳定, 防 A 角漂移污染 abc_offset
   - `DETECT_SKIP_FRAMES` 3→1 (每帧尝试检测)
   - **排坑 (关键 bug)**: `frame_counter % DETECT_SKIP_FRAMES == 1` 在 SKIP=1 时恒为 False (`x % 1 == 0`), `detect()` 永不调用 → 完全无检测结果 (现象: teach_P1 无 [Handle] 输出, 而 handle_cup_detector 独立跑正常, test_skip=5 不受影响); 改为 `== 0` (SKIP=1 每帧触发, SKIP=3 在 frame 3/6/9 触发语义不变)
3. **流程调整 (用户需求)**: ① R0 输入从启动时延后到"把手稳定之后" (先确认检测 OK 再输 R0); ② 显示窗口稳定后关闭, 下次示教迭代重新打开 (imshow 自动重建); ③ R0 首次必输, 之后回车复用上次 (parse_pose_input 加 allow_empty=True)
4. **欧拉约定验证 (用户物理规律)**: `euler_to_rotation_matrix(A,B,C)` → R = Rz(C)@Ry(B)@Rx(A), 即 **A=绕基X / B=绕基Y / C=绕基Z**; `rotation_matrix_to_euler_angles` 返回 (绕X,绕Y,绕Z) 互逆 ✓; **绕基系竖直轴 (Z) 旋转 θ → 只改 C, AB 不变** (Rz(θ)R = Rz(C+θ)Ry(B)Rx(A)), 与用户实机观察 ("水壶正对 XZ 时不同位置抓取 ABC 不变; 未正对时右侧抓取只改 C") 完全一致
5. **推论**: abc_offset 与水壶朝向无关 (水壶绕 Z 转 θ, detect_C 与 P1_C 同增, offset_C 不变) — 跨朝向示教可验证检测器姿态链 C 分量正确性
6. **实机示教跑通 (2026-08-16)**: 数据落盘 `logs/teach_handle_p1_*.log` — 每条示教含时间戳/相机系稳定均值 (cam_mean/euler_cam)/窗口极差 (range pos/eul)/R0/基系偏置 (detect_base/P1/t_offset/abc_offset/P2), 结束含汇总平均+方差 — **memory-bank 不存原始数据** (用户约定, 2026-08-16)
7. **注意**: P1 B=85.1° 接近 ±90° 万向锁区, 欧拉 A/C 分解对噪声敏感 (不影响"绕Z只改C"结论), 若 A/C 方差异常优先怀疑此

### 灵巧手串口 Linux 适配 (2026-08-16)

1. **背景**: Omnihand 系代码默认端口仍是 Windows `COM5`/`COM3` (仅 omnihand_test/enable 已适配 Linux); estun_config.yaml `hand.port` 早已配 Linux by-id 路径, 但代码 fallback 未同步 — 无 config 场景 (voice/speed_control/gesture 独立入口) 会开错串口
2. **本机端口结论**: STM32 VCP 枚举为 `/dev/ttyACM0` (ACM 类, 非 ttyUSB); 稳定路径 `/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00`; 当前用户**已在 dialout 组** (直连, 无需 `sg dialout` 包裹 — 2026-08-14 遗留待办已销)
3. **修改 11 文件 31 处**: `"COM5"`/`--port COM5/COM3` 默认值与 docstring 示例 → by-id 稳定路径: omnihand_gesture / omnihand_speed_control / voice_llm_control / voice_doubao_control / voice_omnihand_control (默认参数+示例), omnihand_test / omnihand_usb_serial (docstring 说明, 禁 ttyACM 编号), state_machine / estun_handle_sm / estun_handle_cup_sm / estun_handle_data_sm (fallback 默认值)
4. **注意**: estun_handle_*_sm.py 构造默认值改后仍会被 estun_config.yaml `hand.port` 覆盖 (运行时以 config 为准); 其他机器按实机枚举修改
5. **验证**: py_compile 11 文件全过; 残留检查仅剩 by-id 说明文字

### putText 中文乱码修复 (Windows→Linux 移植, 2026-08-16)

1. **背景**: OpenCV putText (Hershey 字体) 只支持 ASCII, 中文/✓/↑↓°/均渲染乱码 (Windows 下部分字体可渲染, Linux Qt 后端不行)
2. **修复 3 处**:
   - rps_gesture.py: 手势名 `gesture_cn`(中文)→ 显示用 `gesture_name`(已有英文 CLASS_NAMES); `"✓"`→`"OK"`; `"↑"/"↓"`→`"UP"/"DN"`; `"°"`→`"deg"`
   - estun_teach_handle_p1.py: putText `"把手稳定中..."` → `"Stabilizing..."`
   - estun_teach_p3.py: putText `"Cup 稳定中..."` → `"Cup stabilizing..."`
3. **窗口标题**: 全项目 45 个 namedWindow/imshow 标题已全部 ASCII (之前处理过), 中文标题在 Linux Qt 后端会导致黑屏/打不开 — 新窗口一律 ASCII
4. **验证**: py_compile OK; 全项目扫描无剩余 putText 非 ASCII 字面量

### rps_gesture 相机修正: 补 camera_path (2026-08-16)

1. **问题**: rps_gesture.py 的 `run_test` 创建 UGreenCamera 时漏传 `camera_path` → camera_id=-1 时按名称匹配命中第一个 UGREEN (4.1 口 [6]), 而非 Camera B 的 4.2 口 [8] (hand_in_cup 已传, 实测正确)
2. **修复**: `run_test` UGreenCamera 调用补 `camera_path=_ugreen.get("camera_path", "")`; `_open_camera` 已正确传 path (241 行)
3. **验证**: 编译 OK + find_camera_id 实跑: 路径匹配 "usb-0:4.2" → [8] ✓

### P1 示教工具升级: 6D 偏置 (检测姿态纳入闭环) (2026-08-16)

1. **背景**: 示教 log 只有位置偏置, 检测姿态 (动态 euler) 未参与 — 用户指出姿态应辅助抓取姿态确定
2. **改造 (estun_teach_handle_p1.py)**: 稳定窗口同时收集 pos+euler (只收 source=dynamic 帧); 新增 `transform_pose_to_base_6d` (euler_cam→R_cam→T_cam2ee→R_ee→R_ee_base→基系 ABC, ZYX 约定与 ESTUN ABC 一致) + `mean_euler` (逐分量 ±360k 对齐后平均) + `unwrap_deg` (±180)
3. **偏置语义**: t_offset = P1_teach - detect_base_pos; abc_offset = unwrap(P1_teach_ABC - detect_base_ABC); 运行时 P1_ABC = detect_ABC + abc_offset
4. **输出**: log/打印补 detect_base ABC + abc_offset; yaml 段加 `abc_offset`; 汇总加角度平均+方差 (验收 <5°)
5. **离线验证**: euler↔R 闭环 100 例 / 6D 转换与位置版一致 / unwrap 环绕 / mean_euler 179± / 偏置闭环 全过; 单测 30 例不受影响 (工具独立)
6. **待实机**: 重新示教 ≥3 次, 验证 detect_ABC 合理性 (与示教 P1 ABC 对比) 与 abc_offset 方差

### 波动根因修复: anchor 固定 seed + 系列回退 (2026-08-15)

1. **关键修复: `_get_anchor_depth` 固定 seed** — 随机采样无 seed 时每帧采样 50 点不同 → anchor_depth 抖动 → Z 离群过滤(±5cm)边界变化 → 点云集合每帧不同 → PCA 主轴漂移 (**mask 稳定但姿态大幅波动 44° 的真正根因**; test_handle_6D 原版有 seed(0), 重构时丢失)。修复: `np.random.default_rng(0)` 确定性采样 → 实测 C 角 44°→**3.9°**、B 角 12°→**2.2°**、A 角 14°→8.9° (壶身轴 λ1/λ2=1.14 深度噪声残余); 位置 x/y/z 1-2mm
2. **mask 扩展默认关闭** (extend_px: 0) — L 型(bbox 判定只加右侧段)与补全(x 范围整条)均破坏 PCA 主轴 (Y 角); 中心由 mask_center 保证
3. **中心回退 mask_center** (点云重心) — edge_offset 边界偏移法暂弃: 固定偏移 (down_cm 4cm/in_cm 1cm) 相对 mask 实际尺寸过大 (4cm 超 mask 高→中心到底; 1cm 超 mask 宽→中心偏左)
4. **几何法回退 b0b36cd** (Y=handle PCA 主轴) — "壶身轴×环法线"(min_axis) 试验失败: 实机最小方差方向是**环厚方向** (x 20mm 薄) 而非环平面法线 → B 角大幅倾斜 (-29°); 已删 fallback, 注释记录试验结论
5. z 改进: mask_center 重心 z 实测 2mm (mask 稳定时重心稳); edge_offset 函数保留 (z 点云中位数逻辑) 未用
6. 离线验证: 编译 + 单测 11 例全绿; 实机 30 帧波动大幅收敛 (C 3.9°/B 2.2°/A 8.9°/位置 1-2mm)

### 姿态稳定性: Y 改用壶身轴×环法线 (2026-08-15)

1. **实机数据 (44 帧)**: 位置稳定 (x/y 1-2mm, z 中位数方案后 4mm) 但姿态波动大 — A 14°/B 6.5°/**C 48°** (112.8~160.8); mask 稳定但 6D 波动 → 根因: handle PCA **主轴** (环弧面长轴) λ1/λ2≈3.5 接近, 弧面近对称 → 主轴方向对点云微小变化极敏感 (对角线主方向的本质)
2. **修复**: Y = 壶身轴 × 环法线 — `_pca_axis` 额外返回最小方差方向 (axis_min, 环法线/厚度方向, 物理固定); `_compute_handle_pose_geometry(handle_min_axis, body_axis)` — Y=cross(body, min) 由两个稳定轴决定, 与摆动的主轴无关; 新增 `_compute_handle_pose_geometry_fallback` (退化时 Y=主轴旧法)
3. **排坑 (数学)**: Y = body×min 的水平性依赖"壶身轴≈竖直 + 环法线≈指向相机 (z)" — 侧面视角场景成立 (光轴水平: 壶身轴≈y、环法线≈z → Y≈x 水平); 初版测试用 min_axis≈x 错误 (环厚方向 ≠ 环平面法线)
4. 离线验证: Y 水平 (z≈-0.10)/符号/翻转一致/fallback/主轴摆动无关/单测 11 例全绿; **待实机复测 C 角波动**
5. 中心位置: mask_center (点云重心) 回退生效; edge_offset 边界偏移法暂弃 (固定偏移相对 mask 尺寸过大)

### 检测器重构: 消除与实际不符的代码 + 边界偏移中心 (2026-08-15)

1. **用户指出代码/函数与实际作用不符** → 重构:
   - A: 文件头 docstring 更新 (θ/R_ref 描述 → 3D 几何法 + 边界偏移)
   - B: `_compute_handle_6d_pose` 简化 → 改名 `_compute_pca_axis` (类方法 `_pca_axis`) — 不再构造完整 rot (删 Z=最小方差指向相机约束 + X=Y×Z 重正交化, 消费者只用 Y 列), 返回 `(pos, y_axis, cloud)` 主成分方向; `_compute_handle_pose_geometry` 签名改 `(handle_axis, body_axis)`; 调用处 2 处同步 (行为等价)
   - ~~C: `_extend_handle_mask` 判定修复 (body 中位数边缘)~~ **已回退** — 实机复现"mask 完全不可用", 确认 C 就是上次回退前导致问题的改动之一 (误判根因在 Z 逻辑, 实际是扩展判定); 恢复 a7ecb07 的 bbox 判定版
2. **边界偏移中心法** `_compute_handle_center_edge` (用户方案): mask 覆盖不全时重心不可靠 → 中心 = 上边界向下 down_cm + 把手侧边界向内 in_cm (相机系图像方向实际距离); config `handle_pos_mode: edge_offset | mask_center`, `edge_offset.down_cm: 4.0 / in_cm: 1.0`; 边界像素 <3 回退 PCA 重心; 用户实机看效果中
3. **离线验证**: 编译 + 单测 11 例全绿; 待实机
4. **排坑**: 扩展判定 (body 中位数 edge+1) 在实机导致 mask 完全不可用 — 根因未明 (bbox 判定版正常); 该改动的离线测试 (合成 mask) 全过但实机失败 → 离线合成场景与实机差异大, 改动需实机确认后再保留

### 几何法实机反馈修复 (2026-08-15)

1. **Z 朝向相机修正**: 用户实机观察"Z 似乎没有调整" → 根因: RealSense/OpenCV 相机系 z 正=远离相机, 原实现 `Z[2]>0` 让面法线背离相机; 改为 `Z[2]<0` (朝向相机, 与 test_handle_6D PCA 的 Z 约定一致)
2. **帧间锚定升级 `_align_axes`**: 原 `_align_y_axis` 只修 Y 翻转 (绕壶身轴 180°), 面法线水平时 Z[2]≈0 符号约束失效 → Z 独立翻转无法修正; 改为 4 候选 (恒等+绕 X/Y/Z 各 180°) 取与上一帧 trace 最近者, 覆盖全部翻转模式 (测试: 4 模式全对齐 diff<1e-6)
3. **mask 扩展改为贪心**: 原固定 extend_px=15 + 固定范围 → 核心段偏上/下时弧端超出扩展窗口漏补 (用户: "向下扩展还有空间, mask 下方还有实际把手区域, 中心点偏上"); 改为从核心段边界逐行向外补, 连续 5 行无把手侧像素停止 (弧端连壶身处自然停), extend_px 仅作上限 (50×4); 实测中心 y 183→226 (真实 240), 只补把手侧像素 (x≥bbox_x2-10)
4. Y 方向确认无问题 (用户: "刚才是我看错了")
5. 离线验证: Z 朝相机/4 模式锚定/贪心扩展全覆盖/单测 11 例全绿; **待实机复测**

### 姿态 3D 几何法重构 (用户方案, 2026-08-15)

1. **用户指出 dx/θ 链缺陷**: ①D435i 水壶不动时 dx_ref 自标定永远取不到"正对最大 dx", θ 恒 0, 旋转角确认失效; ②PCA 的 Z (最小方差+"指向相机") 假设把手面正对相机, 但把手是长方体, 抓取必须 TCP 贴面
2. **新坐标系 (用户确认)**: X = 壶身 mask 3D PCA 主成分 (壶身长轴); Y = 把手 mask 3D PCA 主成分 (长轴), 符号按 grasp_direction: right→指向图像下边界 (Y·(0,1,0)>0), left→上边界; Z = X×Y (面法线, TCP 贴合面), 约束 z>0 朝向相机一侧 (不要求完全指向); 右手法则
3. **已实施 (handle_cup_detector.py)**: 删除 dx/θ/dx_ref/R_ref 自标定链 (_extract_handle_features/_estimate_yaw_signed/_update_reference/_synthesize_handle_pose) → 新增 `_compute_handle_pose_geometry` (3D 几何自足, 水壶不动也能定姿态); scanline 严格化 (center_deviation_ratio 0.25, 双向剔, 单行超差即剔) + 新增 `_extend_handle_mask` (bbox 边界版, 上下扩展 extend_px=15 补回弧两端 — 注意: 用 bbox 而非 body mask 做把手侧判定, body 会被未剔除行把手像素污染); `_align_y_axis` 加 axis 参数 (绕壶身轴 180° 锚定, 替代 diag(-1,-1,1) 绕相机 Z — Y 翻转物理上是绕壶身轴); body PCA 每帧计算
4. **排坑**: Y 与图像边界方向垂直时 (点积≈0) 符号判断失效 → prev_rot 锚定兜底 (绕壶身轴 180° 修正, 测试验证翻转输入一致性 diff=0); mask 扩展的把手侧判定用 kettle bbox 边界 (body_x_max 会被扩展区把手像素污染成 415 导致漏补)
5. **离线验证全过**: 符号规则/右手系/Z 朝相机、翻转输入一致性、_align_y_axis 绕壶身轴、mask 扩展只补把手侧; 单测 11 例全绿; **待实机复测 D1-A Task 4** (姿态抖动 <10°)
6. config.yaml: 删 handle_ref 段; scanline 段更新 (0.25/双向剔/extend_px 15); rot_smooth_alpha 顶层

### 抓取流程细化: 180° 翻转修复 + P1 示教工具 (2026-08-15)

1. **180° euler 翻转修复 (3 层根因, 已数值验证)**: ①主导: PCA 主轴符号逐帧任意 + R_ref 未建立期无锚点 (180° 翻转→euler (A,B,C)→(-A,-B,C±180) 与实机 A 聚类精确匹配); ②放大: B≈±90° 万向锁 (0.02rad 噪声 B=89°→317° 偏差 vs B=45°→3.2°, 放大 100 倍); ③次级: _synthesize_handle_pose 的 abs(theta_ema)>0.5 硬切换不连续
2. **修复 (handle_cup_detector.py 内 5 点)**: `_align_y_axis` (frame-to-frame Y 锚定, 锚链 R_ref→prev_rot)、统一合成路径 (去硬切换, θ=0 自然连续)、`_MatrixSmoother` (旋转矩阵 EMA+SVD, 矩阵空间避万向锁)、`_unwrap_euler` (A/C ±360k 最短路径)、B 角监控日志 (lock=|abs(B)-90|); config 加 `rot_smooth_alpha: 0.4`; 失败帧 reset 平滑器
3. **离线验证**: py_compile + 单测 11 例全绿 + 新测试 (Y 锚定幂等/翻转一致、unwrap 179→-179 连续、θ=0.4/0.6 合成连续 0.2°、EMA 正交、翻转帧对齐后与原帧一致) 全过; **待实机复测 D1-A Task 4** (验收: 姿态抖动 <10°, A/B 无符号翻转聚类)
4. **P1 示教工具 `Estun/estun_teach_handle_p1.py` (新增)**: 镜像 teach_p3 骨架 — 只读相机 + 人工示教, 基系偏置 t_offset = P1_teach_base - detect_handle_base (基系直接加, 含 TCP 偏移语义, 无 t_handle_effective 链); 多点平均+方差; --apply 写 estun_config.yaml; 变换链闭环已验证 (任意检测点恒等式成立, 前提 T_cam2ee 准确 + 机器人姿态族一致)
5. **Step 3 (state_machine teach 模式) 用户指示暂停**: 不实施 state_machine 改造/estun_config handle_teach 段; 示教工具可独立先用 (产出偏置等 Step 3 恢复后填入 config)

### D1-A 测量完成 + euler 180° 翻转实机确认 (2026-08-15)

1. D1-A 全部完成: Task 0 安全(控制柜上电, 未使能) / Task 1 相机映射 (UGREEN 4.1 迎宾 + 4.2 桌面 Camera B + D435i 2e:00.3, D435i 固件 serial 342222071364 vs USB 层 208223061472 双序列号已澄清) / Task 2 场景测量 (壶口 0.09/底 0.095/最大 0.13m@0.02, 基准点 [-0.37,0.90,0.295], 相对 x 符号修正; 包络间隙 P1 手-桌面 0.01m/P2 0.02m) / Task 3 标定复核 (替代内参方案全链误差 7~11cm 已记录为方案水平, 相机/标定确认无问题) / Task 4 视觉稳定性 / Task 5 协议 16/16
2. **euler bug 实机确认 (yolo+sam 新链)**: 60/60 检测成功、位置抖动 0.7mm、P2 新鲜观测 0.46s ✅; **动态 euler 帧间 180° 翻转震荡** (A 轴 -100~-178° 与 +130~+169° 跳变, B/C 大幅摆动 ~100°), 即 memory-bank 待办"PCA 主轴 180° 跳变"未完全解决 — 位置稳定姿态不稳, 影响 P1 抓取姿态精度, 修复后需复测
3. D1-B 前风险记录: 全链误差 7~11cm + P1 包络间隙 1cm (碰撞风险, 需低速+急停观察); 材料在 `/data/VLA/VLA-BD/docs/d1-measurements-2026-08-15/`

### Handle 检测回退 yolo+sam + 动态姿态 (2026-08-15)

1. **背景**: 水壶把手红胶带物理移除 → 旧"红胶带优先+白回退"检测链失效; 回退到 RealSense_R 已验证的 yolo+sam (test_handle_6D/test_SAM_handle)
2. **`Orchestrator/handle_cup_detector.py` 重构** (1316 行, 唯一主改造文件; cup 检测与 state_machine/estun SM 未动):
   - 新链: YOLO kettle → SAM bbox 提示 (`model/sam2_b.pt`, 无降级) → mask 后处理 → 扫描线分离 v2 → PCA 6D → 旋转角 θ → 完整动态 euler_cam
   - **mask 后处理** `_postprocess_full_mask`: OPEN 3x3(无损去噪) → ERODE 3x3(消渗漏) → CLOSE 5x5(填洞) → 连通域过滤 (最大域+把手侧孤岛保留, 防 SAM 切孤岛丢把手)
   - **scanline v2** `_scanline_find_handle_v2`: 搜索范围=mask 外接矩形∪bbox (修复 bbox 截断把手外弧); X 中心剔除保守化 (trim_top 默认关保弧, 连续2行超差才剔, 只删对应行)
   - **完整动态姿态** (非固定角度): dx 特征 (把手/壶身 x 中心偏移) + dx_ref 参考 (运行期自标定 max > 几何兜底 body_w×0.45 > config dx_px) → θ=arccos(|dx|/dx_ref) (夹取 50°, EMA 0.4); R_final = Rot(壶身轴 body PCA, θ) @ R_ref (正对参考缓存, PCA 主轴 Y 对齐防 180° 跳变); euler_cam=相机系 ZYX 动态值, 失败帧回落 config 固定值 (契约: 永不为 None)
   - 返回 dict 保留 red_center/white_roi/red_contour 兼容键恒 None
   - config.yaml: 新增 sam_model/sam_conf/sam_roi_expand_ratio/mask_postprocess/scanline/handle_ref/pca_step_handle/pca_step_body; 删除 white_handle/red_handle 段
3. **离线验证**: py_compile OK; 单测 11 例全绿 (7 session_barrier + 4 gate_source, 源码字符串断言通过); 合成 mask 测试: v2 覆盖完整外弧 (xmax 445 vs 旧版分离失败)、后处理噪点清除/洞填上/把手孤岛保留 (ERODE 仅损 5%)、θ→euler 数学单调性验证
4. **性能预算**: 反投影 Python 循环是主要开销 → pca_step_handle=2/pca_step_body=4 (body 仅 R_ref 建立后计算); 单帧预计 ~130-150ms, 实机按终端 ms 调参
5. **待办**: 实机验证 (只读相机): mask+6D 坐标轴 → 后处理效果 → ±30/60° 把手覆盖 → θ 跟随 + 动态 euler (先定 sign 方向); state_machine/estun SM 消费动态 euler 属后续步骤

### 回传材料准备 + skill 包发送 (2026-08-14)

1. `~/p2p1-handoff-staging/` 建立: `git-diff-132051c..HEAD.patch` (13 文件/529 行, 排除 memory-bank 内部记录与对方已有的交接文档) + `handlepluscup-orchestrator.crp.txt` (离线解析版, 143 节点/22 points/嵌套块展开)
2. 基线核对: 对方锁定的 SHA-256 (crp `342cb0e9...` / state_machine `f3cd72b1...`) 与本机当前文件完全一致 — 回传 diff 基线正确
3. 已发送 (对方 inbox/from-real-host/ 全部校验通过): estun skill 包 (sha256 `262cf8db...`) + git diff (sha256 `30d64559...`) + CRP 文本版 (sha256 `45e8f571...`)
4. 已发送 (对方 inbox/from-real-host/ 全部校验通过): estun skill 包 (sha256 `262cf8db...`) + git diff (sha256 `30d64559...`) + CRP 文本版 (sha256 `45e8f571...`) + 运行入口清单 `run-entry-inventory-2026-08-14.md` (main.py `ab13f8...`/state_machine `f3cd72...`/estun_config `d69127...`/crp `342cb0e9...`; 基准 132051c→HEAD 2dd6fc9; CRP 不受 git 管理, 版本=SHA+mtime)
5. 已回复: `/data/VLA/VLA-BD/docs/project-hand-reply-d0-p2p1-materials-2026-08-14.md` 发送对方校验通过 — 含材料 SHA-256 全表 + D0 执行计划 + P2/P1 序列理解确认 + Linux 适配只读证据 (串口 by-id 唯一/VID 0483/PID 5740/当前用户 dialout 组✅; 相机 video0-3 UGREEN ×2 + video4-9 D435i serial 342222071364 USB3.2; 待办: 重登会话 dialout 直连验证)
6. 已发送: 接口协议文档 `project-hand-interface-protocol-2026-08-14.md` (拓扑/Phase0-4步/PhaseA-D序列/位姿格式/超时表/串口帧/安全来源/契约差距) + **CRP 原件** `handlepluscup-orchestrator.crp` (sha256 与对方锁定 `342cb0e9...` 一致)
7. **理解修正**: ".crp 文件本身就是示教器官方工程格式 (单行 JSON)" — 不存在额外"示教器官方导出版", 无需现场导出; 对方 zip 中已有该 crp
8. **新工单 (桌面传入)**: `project-hand-detailed-crp-ticket-template-request-2026-08-14.md` — 要求离线创建动态 ticket CRP 模板 (`p2p1-ticket-string-template-2026-08-14.crp`, 证明 A 字符串拼接/B 精确比较/C 允许拒绝分支/D Socket 超时拒绝 4 能力), 拒收手工拼装 JSON, 不修改生产 CRP。已核验: 本机无 Codroid 编辑器 (Linux), 现有 11 CRP 无字符串拼接/比较 block 实例 → 模板须在 Windows 开发机/示教器编辑器上创建
9. 已回复: `/data/VLA/VLA-BD/docs/project-hand-reply-crp-ticket-template-2026-08-14.md` 发送对方校验通过 — 确认工单+禁止事项, 澄清请求 3 项 (编辑器位置/版本读取/回传方式), 附执行计划; 等对方答复
10. **session-barrier 工单已完成回传** (2026-08-14): 对方以《project-hand-estun-session-bound-barrier-work-order-2026-08-14.md》取代动态 ticket 模板工单 (原工单+环境决定作废, 仅"匹配版本 Windows 编辑器/不操作实机"限制保留)。采用现有 block 能力 (CPOS 字符串化/SocketReadStr/TranStrToCpos/数值 flag/If/SocketClose/Return) 实现 P2/P1 双屏障: 生成 `doc/handlepluscup-orchestrator-session-barrier-2026-08-14.crp` (207262B, SHA `3d16aa59...`, 生成脚本 `scripts_gen/gen_session_barrier_crp.py`, 全部 block 从源工程克隆, 零新增 MovL, DummyGraspAck/DummyGripAck 零运动引用), 编辑器 v1.6.3b 打开验证通过, manifest 已生成, 三件套已发对方校验通过
11. **session-barrier v1 被审计退回 + v2 返工已回传** (2026-08-14): 退回原因 6 处字段不一致 — 4×`lengthunit=mm`→`m` (CPosToStr/TranStrToCpos 克隆自 Phase 0 模板)、2 组 send/read `Socket1`→`Socket0`。已修生成脚本 (显式 patch lengthunit=m + mscname=Socket0) 生成 v2 (`doc/handlepluscup-orchestrator-session-barrier-v2-2026-08-14.crp`, 207258B, SHA `1e109d57...`), 全验证通过, 编辑器 v1.6.3b 打开验证通过, manifest v2 已生成, 三件套已发对方校验通过 (不放桌面, 按要求只放 VLA-BD/docs + inbox)
12. **Python host session gate 已实现并回传** (2026-08-14): 对方 v2 验收通过后指令. 新增 `Orchestrator/session_barrier.py` (SessionBarrier 状态机 WAIT_P2_ARRIVED→WAIT_P1_ARRIVED→WAIT_GRIPDONE→PASSED + pregrasp/grasp 一次性 ticket + 严格 CPOS 解析 + 容差/时效/标定摘要/新鲜观测校验 + BarrierReject 原因码), `state_machine.py` 改造 (load_cam2ee 严格化抛 CalibrationError 禁 np.eye(4) + 标定加载移至 TCP server 前 + 保留 D435i 至 P2 复核 + _refresh_handle_observation/_release_camera_a_late + GRIPDONE 一次性 `_grip_done_once`), `handle_cup_detector.py` +last_known_pose_ts, 单测 22 例全过 + py_compile OK. 文件已回传 `/data/VLA/VLA-BD/docs/python-session-gate-2026-08-14/` + 对方 inbox 校验通过 (SHA 见 changelog)
13. 待办: 等对方 source pin + 静态状态顺序/标定回退/相机生命周期/hand-close 调用链审计 → 通过后请求故障注入回放/D1; 本地未提交改动 (state_machine/handle_cup_detector/session_barrier/tests) 待对方审计通过后一并提交

### 训练侧 D0 门禁 + P2/P1 双确认请求 (2026-08-14)

- 对方回传两份请求（`/data/VLA/VLA-BD/docs/`）：D0 执行门禁 + 控制器 P2/P1 双确认协议适配
- 已审计：标定文件合格（Daniilidis/26 样本/0.228px）；15 个 py 文件含 np.eye(4) 回退待清理；现 CRP `H_P2→H_P1→GRIPDONE` 不满足腕部重观测约束
- 要求：标定失败阻止启动、PREGRIP 仅手势、双一次性确认点（ticket 绑定、超时/断连安全停止）
- 待回传：CRP 文本版、`git diff 132051c..HEAD`、运行入口+CRP 版本、SHA-256
- 交接文档：`doc/p2p1-dual-confirm-handoff-2026-08-14.md`；CRP skill：`.claude/skills/estun-omnihand-robotics/`

### Linux 环境迁移适配 (2026-08-14)

1. **`EyeToHand/camera.py`** — 跨平台化:
   - Linux: `CAP_V4L2` + 先设 MJPG 再设分辨率 (720P/1080P 必需,参考 `Project/camera_view.py`),无需 pygrabber
   - Linux 设备名探测: 遍历 `/sys/class/video4linux/videoN/name` 匹配 "UGREEN Camera 1080P" (替代 pygrabber DSHOW)
   - Windows 原 DSHOW + pygrabber 逻辑保留, `_IS_LINUX` 分支切换
2. **依赖安装 (Proj 环境, Linux)**: opencv-python 4.13.0.92 / PyYAML 6.0.3 / pyserial 3.5 / pyrealsense2 2.56.5.9235 / ultralytics 8.4.48 / mediapipe 0.10.35;torch 2.13.0+cu126 已有
   - 排坑: PyPI 上 pyrealsense2 版本带构建后缀 (`2.56.5.9235`),requirements 的 `==2.56.5` 匹配不到
   - 排坑: mediapipe 拉入 opencv-contrib-python 5.0.0.93 覆盖锁定版本 → 降回 4.13.0.92 (cv2 与 requirements 一致)
   - 按需求排除: EpicEye SDK / Orbbec / pygrabber / comtypes / 语音控制包
3. **清理**: `__pycache__` 与 `.pyc` 全清; `reference/机械臂_同事运行包_20260713/` 下 Windows 专属二进制删除 (pyrealsense2 win .pyd + RealSense SDK 安装器 x3);logs/runs/calib_data/model 保留
4. **git 换行归一化**: 仓库 blob 为 LF、Windows 工作树为 CRLF (83 文件) → 设置 `core.autocrlf input` 后 86 个假变更归零; `core.ignorecase=true` (Windows 克隆遗留) 掩盖过 doc/ 两个文件大小写改名 → 本次提交暴露为 rename
5. **`.gitattributes`** — 新增 `* text=auto` + 二进制类型标记, 换行策略固化进仓库

### 三相机联调验证 (2026-08-14)

- 三相机同时采集: UGREEN x2 (1280x720 MJPG, 27/28fps) + D435i (640x480 color+depth, 28fps) 全速无冲突
- **排坑: /dev/videoN 编号不稳定** — 插拔 USB 和 pyrealsense2 驱动 unbind/rebind 都会重排 video 节点; 硬编码索引 (如 video6) 会开到错误的设备, 表现是 "Device or resource busy" 或 "can't open camera" (曾误判为 USB 控制器等时带宽争抢)。**结论: 一律用 `/sys/class/video4linux/videoN/name` 按名称枚举 (EyeToHand/camera.py 已实现), 禁止硬编码索引**
- `multi_camera_preview.py`: 名称枚举版三相机预览 (UGREEN 自动探测 + D435i color/depth), q/ESC 退出清理正常
- **花屏根因 (已解决)**: video0 的 "Corrupt JPEG" 是 D435i 与 UGREEN 同 hub 时的等时带宽饱和 (isoc 丢包, 内核日志无报错, 单独跑或降带宽即消失)。**修复: D435i 挪到独立 USB 控制器** (现 2e:00.3), UGREEN x2 在 2a:00.1 — 三相机 720p30 全速 0 丢包
- 现场结论: D435i (无压缩 color+depth, ~46MB/s) 必须与 UGREEN 分开 hub/控制器; 两台 UGREEN 同 hub 720p30 无问题
- **Camera B 固定**: 两台 UGREEN 名称/序列号完全相同 (假序列号 SN0001), 唯一区分手段是 USB 物理口 → `camera_path: "usb-0:4.2"` 写入 `EyeToHand/config.yaml`, `camera.py` 支持 ID_PATH 匹配 (优先级高于名称), person_detector/rps_gesture/hand_in_cup 已接入
- **cv2 Qt 后端中文窗口标题 → 全黑窗口**: Qt5 highgui 字体配置缺失时, 中文标题窗口内容渲染全黑 (白图正常、帧采集正常); 窗口标题必须用 ASCII (如 "UGREEN 4.1" 而非 "UGREEN 口4.1")

### 豆包语音控制 (2026-08-13)

1. **`Omnihand/voice_doubao_control.py`** — 新建(单文件实现)。豆包 Seeduplex 全双工 S2S:
   - WS `wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue`,X-Api-Key,纯 JSON 文本帧,model 固定 `1.2.6.1`
   - 输入 pcm 16k int16 20ms/包 Base64;输出 pcm_s16le 24k 流式;服务端 VAD(无需本地 Silero)
   - 手势动作走 **Function Calling**(3 工具: do_gesture/do_sequence/stop_hand),入队立即回传,客户端硬校验
   - 4 线程: Main + PortAudio 回调 + WS send(唯一发送者/重连)+ WS recv(事件分发)+ 手势执行线程(串口独占)
   - StreamingPlayer(sounddevice 24k,预缓冲 3 块)替代 Windows MCI;barge-in(response.cancel + epoch 丢弃)
   - speed 三档真正落地: null 直接 / fast 0.75s 插值 / medium 1.0s / slow 3.0s
   - **无降级**: 无 key/连接失败/播放失败/灵巧手失败一律显式报错退出;鉴权 4xx 不重试;5xx 指数退避重连(上限 5 次)
2. **`Omnihand/voice_llm_prompt.py`** — 追加 DOUBAO_INSTRUCTIONS + DOUBAO_TOOLS + validate_tool_call + parse_fc_items(现有 MiMo 部分一字未动,两条链路共存)
3. **`Omnihand/voice_llm_config.json`** — 追加 `doubao` 段(api_key 空占位;真 key 由用户写入 .local.json)
4. **`Omnihand/tests/mock_doubao_server.py` + `test_voice_doubao_offline.py`** — mock WS 服务器(websockets)+ 18 用例离线测试全通过
5. **`requirements.txt`** — +websocket-client==1.8.0(运行时);websockets 仅 mock 测试用
6. **`doc/voice_control_design.md`** — 追加附录 C(协议要点、与 MiMo 对比、工具定义、测试结果)

### 关键协议细节(排坑记录)

- websocket-client 帧传输中途 recv 超时会损坏数据流 → 连接后 `settimeout(None)` 无限阻塞,死链靠 ping + 服务端关闭检测
- 下行事件名带前缀: `response.function_call_arguments.done`(不是 function_call_arguments.done)
- 优雅关闭: session.close → 等 session.closed → 再断连(防 55000001)
- mock 服务器 shutdown 需 asyncio.Event 通知 serve 退出,不能用 loop.stop() 硬停(pending task 告警)
- **全双工必须持续推流** — commit 后停止发包触发 `52000033 AudioServerNoAudioInputTooLongError`;主程序 auto 模式麦克风常开天然规避
- **tools 必须扁平格式** — `type=function` 与 `name/description/parameters` 同级;OpenAI 式 `function` 包装服务端不识别,FC 永不触发(真实 API 探测验证)
- **audio.delta 包大小不固定** — 实测远大于 20ms;StreamingPlayer 已加跨块余量缓冲(不能按一包一块丢尾部)

### 真实 API 验证 (2026-08-13)

- API key 已存入 `voice_llm_config.local.json` (doubao.api_key)
- 免麦克风探测 (临时脚本, 已删除;zh.wav + MiMo TTS 合成语音): WS 连接 ✅ 服务端 VAD+ASR ✅ LLM 对话 ✅ TTS 音频流 ✅ FC(扁平 tools)✅ 优雅关闭 ✅
- FC 实测: "握拳" 语音 → `do_gesture(FIST)` 触发成功, tool 回传后模型语音回复 "好的,已经握拳啦"

### 实机联调 (2026-08-13 下午)

- 实机"没回应"根因: **`_mic_callback` 写 `_mic_q` 但 send 线程只消费 `_send_q`, 麦克风数据从未上送** (离线测试手动塞 `_send_q` 未覆盖真实数据流路径). 已修: send 循环每轮 `_drain_mic` 持续推流 + 5s 电平统计日志
- 实机验证: "你好" → ASR 实时转写 → FC do_sequence(打招呼) → LLM 回复"你好呀!" 全链路 ✅
- 麦克风: 默认设备=英特尔智音阵列(MME index 1), 安静时输出数字零属正常; 环境噪声电平 ~1200-2200 偏高, 实机联调注意降噪/麦克风距离
- 教训: **离线测试全绿 ≠ 实机路径通** — 关键数据流路径必须实机跑一遍

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | **实机验证 band 输出** | 跑 handle_cup_detector.py 独立测试, 各朝向观察 ROT band/th 显示符合预期 (尤其 +45 需 u≥530) |
| 🔴 | **C 补偿接入抓取** | θ/band 已由检测器输出, C=C0-θ 接入 estun_handle_data_sm.py P1 姿态 (A/B 恒定) |
| 🟡 | 15° 档补数据 | 15° 档 LOOCV 仅 2/4（带宽 0.013<间隙 0.0086-0.0093）；补 3-5 位置或并入相邻档 |
| 🟡 | 多朝向示教 | 旋转检测就绪后：各朝向实机抓取验证 → 位置/姿态落点对比 |
| 🟡 | CRP 实机导入测试 | estun_handle_sm.crp + estun_handle_cup_sm.crp（此前遗留） |
| 🟡 | 单相机 SM 实机联调 | estun_handle_sm.py + estun_handle_cup_sm.py（此前遗留） |
| 🟡 | 豆包实机联调（阶段 1/2/3） | 无手对话 → 空载手势 → 握手/急停/抢话/断网/音色 |
| 🟡 | MiMo vs 豆包延迟对比 | 各 5 次，结果写回附录 C |
