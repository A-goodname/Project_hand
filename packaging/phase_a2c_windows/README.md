# Phase A–C 水壶倒水 | Windows 独立运行包

本包识别水壶把手和水杯，抓壶、倒水、放回水壶并回 Home。水杯留在原位。抓壶和倒水沿用实机已验证的 A2D 参考段，控制器使用原有单向指令和固定等待。

## 安装

解压后在 `PhaseA2C-Windows` 目录打开 PowerShell。使用 Windows x64、Python 3.11 x64。已有可用的 Proj 环境时，请先执行依赖安装命令。新环境按以下步骤安装。CUDA 12.6 为打包机使用的 PyTorch 版本。

```powershell
conda create -n PhaseA2C python=3.11 pip
conda activate PhaseA2C
python -m pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r ./requirements.txt
```

RealSense 由 requirements.txt 安装，版本为 `pyrealsense2==2.56.5.9235`。请解压到新目录，避免旧版包根目录的 RealSense `.pyd` 优先于安装的 Python 包。请保留目录结构。两个 YOLO 权重及对应的手眼标定均已包含。首次安装 Python 依赖需要网络。

## 配置与启动

- `Estun/estun_config.yaml` 保存本机监听地址 `192.168.101.96:8080`、灵巧手 `COM4` 和标定文件路径。控制器通过 CRP 连接该地址。
- `RealSense_R/D435i_info.yaml` 保存 D435i 序列号 `342222071364`，换设备时使用该设备匹配的配置与标定。
- `Orchestrator/config.yaml` 保存把手、水杯、判稳及倒水位参数。现有参数来自打包时的 Windows 工作区。
- 参考融合入口也占用 8080，启动前请退出它。

```powershell
python -u .\Orchestrator\estun_phase_a2c_handle.py
```

看到 `Listening 192.168.101.96:8080` 后，将机械臂置于原 A2D 示教的检测起点并运行控制器程序 `doc/controller_exports/PhaseA2C-HandleOnly.crp`。控制器上报当前位姿，Python 等待把手和水杯稳定并发送 H_P1、H_P2、H_P3。CRP 计算 H_P4 后执行倒水与回撤。H_P3 是倒水位，H_P4 是倒水前后的过渡位。回 Home 后控制器关闭连接，Python 退出。

命令末尾加 `--dry-run` 可只预览流程。实际运行日志位于 `logs/phase_a2c_handle_*.log`。

## Claude Code 技能

随包附带 `.claude/skills/estun-omnihand-robotics`，包含技能说明、CRP 辅助脚本和参考文档。
