# Windows：Phase A–C 水壶抓取与倒水

专用版使用 D435i 识别水壶把手和水杯。水杯位置用于计算倒水点。流程完成后水杯留在原位。

在项目根目录用 PowerShell 启动：

```powershell
& 'D:\Environment\miniconda\envs\Proj\python.exe' -u '.\Orchestrator\estun_phase_a2c_handle.py'
```

要预览流程，可显式传入 `--dry-run`。专用版和融合入口都使用 TCP 8080，请单独启动。看到 `Listening 192.168.101.96:8080` 后，将机械臂置于原 A2D 示教的检测起点并运行 `doc/controller_exports/PhaseA2C-HandleOnly.crp`。控制器启动时上报当前位置。

1. Phase A 接收末端位姿。Phase B 等待把手和水杯稳定，沿用 A2D 算法发送 `<H_P1>[H_P2]{H_P3}`。
2. 控制器从 H_P2 接近 H_P1，在 H_P1 发送 GRIPDONE。控制器用原有 CposOffset 和 Retreat1/Retreat2 计算 H_P4。
3. 沿用 H_P4 → H_P3 → H_P4 → H_P1 的轨迹，原版在 H_P3 停留 3 秒。返回 H_P1 后发 RESET，再退到 H_P2 并回 Home。
4. 回 Home 后关闭 Socket。手势采用原版单向指令和固定等待。Python 记录串口异常，日志位于 `logs/phase_a2c_handle_*.log`。

重新生成 CRP 时，用 Proj Python 执行 `scripts_gen/gen_phase_a2c_handle_crp.py`。输入是 `PhaseA2D-KeyboardGate.crp`。生成物需在现场验证控制器导入与轨迹。

## 独立运行包

在项目根目录运行以下命令。生成器使用本机的控制器导出、模型权重、标定和 ESTUN/OmniHand Claude 技能。如果默认输出文件已存在，可以用 `--output` 指定新文件名。

```powershell
python -B .\scripts_gen\gen_phase_a2c_handle_crp.py
python -B .\scripts_gen\package_phase_a2c_windows.py
```

默认输出为 `reference/PhaseA2C-Windows.zip`。包内的 README 和依赖来自 `packaging/phase_a2c_windows/`。RealSense 通过 `pyrealsense2==2.56.5.9235` 安装，版本与原 Windows `.pyd` 一致。

## 第一轮数据采集

设备静止复测、整轮采集范围和待接入项见 [Phase A–C 倒水数据采集](phase_a2c_data_collection.md)。
