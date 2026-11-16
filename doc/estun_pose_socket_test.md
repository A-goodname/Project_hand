# ESTUN 实际位姿 Socket 试验

本项目只读取当前 CPOS 和 APOS、格式化报文并通过 Socket 发送。控制器不执行运动。PC 一侧只收数据。

在 PowerShell 启动：

```powershell
& 'D:\Environment\miniconda\envs\Proj\python.exe' -u '.\Orchestrator\estun_pose_socket_test.py' --with-joints
```

看到 `[LISTEN] 192.168.101.96:8810` 后，在控制器导入 `doc/controller_exports/Estun-Pose-Joint-Socket-Test.crp`，运行 `main1`。控制器读取十组当前 CPOS 与 APOS，逐组编号 `0001` 至 `0010`，每组之间等待 250 ms。报文示例为 `P|0001|[0.123,0.234,0.345,10,20,30];`。位置单位为 m，姿态单位为 deg。静止时位姿重复是正常现象。

Python 写入 `logs/estun_pose_test_*.jsonl`，包含编号、实际 CPOS、原始报文和 PC 的 UTC／单调接收时间。控制器采样时间字段为 null。TCP 可能把多组报文合在一次读取中，这些报文共用该次接收时间。只在两路都收到完整、有序的十组且控制器正常关闭连接时报告 `complete=True`。

CRP 同时附带 `PoseSocketSample` 程序。示教器把当前模板类型显示为「含运动指令的程序」，内部只有读位姿和 Socket 测试块。两个程序都引用主任务点表里的 `PoseTestMain`，对应示教器在该程序中提供的点位选项。每次只由一个任务发起连接。并行测试需要从示教器的「+副任务」创建实际的副任务并导出含 RUN 的模板。静止双路测试已用 GetCurAPos 和 APosToStr 验证。

## 六关节补测

`--with-joints` 要求两路数据。控制器每组用相同的编号发出当前 TCP `P|0001|[x,y,z,a,b,c];` 和当前关节 `J|0001|[j1,j2,j3,j4,j5,j6];`。关节角以 deg 上报。接收端分别记录两帧的 PC 接收时间，两个采样动作有先后，设备时间留空。只有两路各完整、有序地收到 0001 到 0010 才算通过。

配套试验文件为 `doc/controller_exports/Estun-Pose-Joint-Socket-Test.crp`。导入后检查 `main1` 的 GetCurAPos 和 APosToStr 是否都绑定为 `PoseTestJoints`，并执行工程校验。2026-11-16 静止实测已收到 10 组同编号 TCP 和六关节角，共 20 帧，无缺号或解析错误。日志为 `logs/estun_pose_test_20261116_092258_041216Z.jsonl`。静止运行 `main1` 即可测试，真正的并行 RUN 测试仍需实际副任务的导出模板。需要单独复测 P 帧时，可用生成器的默认模式重新生成 TCP 单路 CRP。

已保留通过实测的 CRP。需要重建时，在项目根目录执行 `python -B scripts_gen/gen_estun_pose_socket_test_crp.py --with-joints`。生成器需要 `doc/controller_exports/PhaseA2D-KeyboardGate.crp` 及本机的程序导出模板，可用 `--sub-template` 显式指定后者。
