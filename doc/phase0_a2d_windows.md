# Windows Phase0 + PhaseA2D 联合运行

当前联调版本以控制器导出的 CRP 为基线；`doc/controller_exports/PhaseAll-NoRPS.crp`
是 `D:/Download/PhaseAll-Down-617.crp` 的派生件，不含 PhaseE/RPS。
`Phase0-Down-965.crp` 与 `PhaseA2D-Down-444.crp` 的原样副本也在同目录。
源文件 `D:/Download` 未改动。

## 已核验

- Phase0 迎宾相机：`UGREEN Camera 1080P`，本次 DSHOW 索引 4；抓帧和 YOLO 人检测调用正常。
- PhaseD 判断相机：`UGREEN Camera`，本次索引 3；抓帧与 HandInCup 单帧推理正常。
- 灵巧手：COM5 初始化及 RESET 正常；D435i 序列号 `342222071364` 已枚举。
- 控制器 CRP 连接 Python 本机 `192.168.101.96` 的 8600、8080 端口；机械臂 IP 为 `192.168.101.100`。
- `Orchestrator/config.yaml` 保留服务器未提交的加速参数 `skip_frames: 2`、`stabilization.window: 6`。
- PhaseD 键盘决策：`C` 或回车为 CONTINUE，松夹并发送 `<C_P3>`；CRP 等待 1 秒后 MovL Home。`A` 为 ABORT，先保持夹持并发送 `[C_P3]`；CRP 按 MovL C_P1 → RELEASE → 等 1 秒 → MovL C_P2 → RESET → 等 1 秒 → MovL Home 执行。视觉仅提示，不自动决定。
- 用户已实机确认独立 Phase0 全功能正常，以及独立 A2D 的 ABORT 分支正常；融合版复用同一 PhaseD 决策与 CRP 分支，但 Phase0→A2D 交接仍待实机验证。
- 用户随后实机确认融合版 CONTINUE 分支通过；融合 ABORT 尚待验证。入口运行时自动保存主进程和迎宾相机子进程日志到 `logs/phase0_a2d_*.log` 与同名前缀的 `_phase0_camera.log`（目录被 Git 忽略）。
- 2026-09-23 再次实机通过融合 CONTINUE；本机日志 `phase0_a2d_20260923_115143_084512.log` 显示 Phase0→A2D 共用 COM5、Cup/Handle 判稳、D435i 关闭、键盘 CONTINUE 松夹与收尾均正常。Cup 在 6 个有效样本后稳定；Handle 位置极差约 1–2 mm、姿态极差 <0.5°，主要等待 θ 极差从约 5–12° 收敛到 4.95°（阈值 <5°）。不宜凭本次结果直接放宽抓取阈值。
- A2D 启动时可预开 PhaseD UGREEN、加载 HandInCup 模型；READY 前不读取/推理，READY 后丢弃 3 帧缓冲画面再推理。PhaseB 发送位姿后立即关闭 D435i pipeline。

## 启动

在项目根目录运行（主进程始终持有同一 COM5 连接；迎宾 UGREEN 与识人模型在独立子进程中打开、预热，并在识人后退出；D435i/Handle+Cup 模型默认提前打开）：

```powershell
D:\Environment\miniconda\envs\Proj\python.exe -u Orchestrator\estun_phase0_a2d.py
```

等待 8600 和 8080 服务就绪后，由现场操作员在示教器上导入并运行
`doc/controller_exports/PhaseAll-NoRPS.crp`。先空载、低速验证路径，急停旁有人值守。
PhaseD 到 READY 后看相机辅助画面，在相机窗口按 `C`/回车或 `A`；也可在终端输入
`c`/`a` 再回车。`--no-preview` 仅使用终端输入。
若不想提前打开 D435i，可加 `--no-prewarm`，但控制器连接到 8080 后会多等模型与相机初始化。
Handle/Cup 判稳慢时先看主日志的 `[STABILITY]` 行：有效 θ 样本数、位置/姿态/θ 窗口极差会指出哪个条件尚未满足。当前 `skip_frames=2`、稳定窗口 6 个有效检测样本；Handle 另要求位置极差 <10 mm、姿态极差 <5°、θ 极差 <5°，θ 最新观测不超过 1.5 秒。无效帧不计入窗口，并不会清空已有样本。不要仅为加速而降低抓取门槛。

单独测试 PhaseA2D：

```powershell
D:\Environment\miniconda\envs\Proj\python.exe -u Orchestrator\estun_handle_cup_sm.py
```

服务显示监听 `192.168.101.96:8080` 后，在控制器运行 `doc/controller_exports/PhaseA2D-KeyboardGate.crp`（57 个顶层块，保留控制器原 A2D 运动路径，仅增加 PhaseD 键盘分支）；不要使用原始 `PhaseA2D-Down-444.crp` 来验证 ABORT 分支。

未收到人工决策时 Python 不会松夹、也不发送完成信号；CRP 读超时后应由现场操作员处置。
ABORT 信号发出时仍夹持，直到 CRP 到达 C_P1 发 RELEASE；若分支途中报错/中断，程序不会在退出时自行 RESET，请先人工确认杯、手和机械臂状态。

## 尚未完成的实机验证

- 独立 A2D 的 CONTINUE 分支和融合版 ABORT 分支尚待验证；融合版 CONTINUE 已由用户实机确认通过。
- `.crp` 已在 Git 中忽略，仅保留在本机。换机器时重新取得控制器导出件、原参考 `doc/handlepluscup-orchestrator.crp`，运行 `scripts_gen/prepare_controller_crps.py` 生成派生件。
- 如 Windows 尚未配置 `192.168.101.96`，服务无法绑定该地址；不改项目中的 host，先由现场确认本机网络设置。
