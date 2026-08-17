# Active Context — Project_hand

> 工作区 /data/VLA/Project_hand(2026-08-16 起唯一;~/ 旧副本仅历史)。

## 当前焦点

**data_sm 接入 handle_teach（本地主线）** — 正对示教验收全过后（8-17 log: t_offset 方差 2mm/abc_offset 2.5°），把示教偏置接入运行时完整流程（抓壶→倒水→放回→Home），落点 = `estun_handle_data_sm.py`（无 Cup 阶段 + cup 检测定 P3，CRP `estun_handle_data_sm.crp` 现成）。代码+离线数学验证完成（log 回放 P1 误差 0.7-3.2mm）。**下一步:实机只读跑 data_sm 看 Handle/Cup STABLE → P1 落点对比（正对场景）→ 通过后 `handle_teach.enabled: true` 完整流程低速验证**。旧焦点（实机示教 P1 正对验收）已完成；state_machine teach 模式（Step 3）仍暂停未实施。

训练侧线（并行）：**P2/P1 双确认协议适配（对方 D0 门禁）** — 审计 132051c 后要求：标定失败阻止启动、PREGRIP≠位姿确认、P2 后腕部重观测 + GRASP_AUTH/GRIP_AUTH 一次性双确认。协作线交接见 `/data/VLA/VLA-BD/docs/handoff-vla-collaboration-2026-08-15.md` + `/data/VLA/Project/collab-line/`（不随本仓库）；D1-B 阻断项离线修复中（`/data/VLA/VLA-BD/docs/d1-a-return-audit-and-d1-b-blockers-2026-08-15.md`）。

## 最近变更

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
| 🔴 | **data_sm handle_teach 实机验证** | `conda run -n Proj python Orchestrator/estun_handle_data_sm.py`（只读，不连机器人）：①确认 [Handle STABLE]/[Cup STABLE] 输出 ②机器人连上后 Phase B 发送前看 [TEACH] 打印 P1，与示教 log P1 对比（正对场景）③落点 OK → `estun_config.yaml` `handle_teach.enabled: true`（固定 ABC 变 log p1_abc）④完整流程低速验证（CRP `doc/estun_handle_data_sm.crp`，DCp1 回退避碰 + 放回 + Home） |
| 🟡 | A 角残余 8.9° | 壶身轴 λ1/λ2=1.14 深度噪声放大——可选 `rot_smooth_alpha` 调低（0.4→0.25）或壶身轴与图像竖直混合（HSV 链已消除，留存参考） |
| 🟡 | 多朝向示教 | 正对完成后：水壶旋转至其他朝向示教多位置 → abc_offset/detect_C 变化曲线（HSV 链 C 系统偏差验证） |
| 🔴 | Linux 实机验证: UGREEN | V4L2 720P MJPG 打开 + PersonDetector 全链路跑通 |
| 🔴 | Linux 实机验证: 机械臂链路 | estun_config.yaml 按 Linux 调整 (IP/串口 COM5 → /dev/ttyUSB*) |
| 🔴 | 豆包 API Key 申请 | 火山引擎控制台 → API Key 管理,写入 .local.json 的 doubao.api_key |
| 🔴 | 实机阶段 1 | 无手纯对话: ASR 正确性 + TTS 流式出声 + 首包延迟记录 |
| 🔴 | 实机阶段 2 | 空载: 单手势/速度三档/序列/自定义多手势/越界拒绝 |
| 🟡 | 实机阶段 3 | 握手压力检测、stop 急停、抢话打断、断网重连、音色选择(4 选 1) |
| 🟡 | 延迟对比 | 同句指令 MiMo vs 豆包各 5 次,结果写回附录 C |
| 🟡 | CRP 实机导入测试 | estun_handle_sm.crp + estun_handle_cup_sm.crp(此前遗留) |
| 🟡 | 单相机 SM 实机联调 | estun_handle_sm.py + estun_handle_cup_sm.py(此前遗留) |
