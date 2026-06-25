# Active Context — Project_hand

## 当前焦点

back 模式 P3 示教验证通过 + 抓取微调完成 + B/C 角防跳变修正 (仅 vertical 模式).
P4 预倒水点待机械臂端实现 (CposOffset).

## 最近变更

1. **back 模式 P3 示教验证通过**: 实际倒水精度验证, 运行 demo/run 测试确认可行
2. **t_handle_offset / t_cup_offset 微调完成**: 用户自行在 `estun_config.yaml` 中调整
3. **B/C 角防跳变修正** (2026-06-23):
   - `estun_handle_cup_demo.py` / `estun_handle_cup_run.py`: Handle P1 B/C 角 + Cup P1 B/C 角统一为 vertical 模式守卫
   - `estun_handle_demo.py` / `estun_handle_run.py`: Handle P1 B/C 角统一为 vertical 模式守卫
   - 策略: vertical 模式下 B 角 `<90->89, >90->91`, C 角 clamp 到合理范围; free 模式仅可达性修复, 不修正角度
   - 注释同步更新: 标注 "仅 vertical 模式; free 无限制, 仅可达性修复"
4. **OpenCV 窗口标题去中文**: 去掉 `"新模块"` 中文, 消除 Windows GBK 乱码导致的双窗口问题
5. **`doc/next_steps_prompt.md` 删除** — 已读取并清理解除

## 下一步

- P4 预倒水点: 由机械臂端 CposOffset 实现, 定义 `[P1.x, P1.y, P3.z, P1.a, P1.b, P1.c]`, 路径 `P2→P1→P4→P3→P4→P1→P2`
- left 模式示教 + t_cup_offset_left 微调 (待完成)
- Orbbec 相机与主入口集成
- RealSense (新版, rs.align) 模块与主入口集成
- 双相机体系 (眼在手上 + 眼在手外) — 见 doc/Future_work.md
- 全面自动化测试
