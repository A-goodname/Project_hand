# Active Context — Project_hand

## 当前焦点

Handle P1 角度范围限制: 对 Handle（右手抓取）P1 点的 B 角新增 range clamp (80,90) → 85.

## 最近变更

- 在 `Estun/estun_handle_demo.py`, `estun_handle_run.py`, `estun_handle_cup_demo.py`, `estun_handle_cup_run.py` 共 4 个文件中的 Handle P1 计算处（vertical/free 模式），C 角 clamp 之后新增 B 角 clamp：
  - 范围: `[80, 90]`, 超出则赋中值 `85`
  - fixed 模式不受影响（直接使用配置的 FIXED_ABC）
  - Cup 部分不受影响（已有独立约束）