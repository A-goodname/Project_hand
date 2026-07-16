# Active Context — Project_hand

## 当前焦点

**人就位 GPU 优化 + Cup P3 速度优化 完成 ✅**

## 最近变更

### v13 人就位 GPU 化 + Cup sudu2 提速 (2026-07-16)

1. **PersonDetector GPU 化**:
   - `EyeToHand/config.yaml`: `device: cpu → gpu`
   - 合并 person+chair 双 YOLO 调用为单次推理 (`_detect_person_and_chair()`)
   - `leave_frames: 30 → 45` (配合 GPU 高帧率)
   - 旧方法保留, 独立测试模式仍可用

2. **Cup P3 速度优化 — sudu2**:
   - CRP 新增 `sudu2` 变量: sper=55, stcp=150, sori=65
   - 4 处 MovL sudu→sudu2: C_P5↔C_P3 递送 + ABORT C_P5/C_P1 退避放回
   - 倒水 H_P3/H_P4 保持 sudu (壶重+倾斜)

3. **Skill 更新**:
   - CRP 结构快速分析 3 步法
   - 编码陷阱: bash 中 `open()` 必带 `encoding='utf-8'`
   - 变量多容器遍历: `projectcfg.vars.data` 所有子分类 + `data.vars.project`
   - 备份规则: 后缀 `.bak` 不加时间戳

4. **CLAUDE.md 全局规则**: 新增第 8 节"搜索与推理纪律"(空结果≠不存在/信息不全禁行动/锚定/回顾已知规则)

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🟡 | 加水实测 | 完整流程加水倒水, 验证防溅 + HandInCup 判定 |
| 🟡 | 抓取姿态微调 | 末端抓取姿态 + 灵巧手手势微调优化抓取成功率 |
| 🟢 | 实机验证 sudu2 | CRP 导入示教器, 验证 sudu2 速度合理性 |
| 🟢 | 完整流程稳定性测试 | 多次连续运行验证 |
| 🟢 | MediaPipe clearcut 彻底抑制 | shutdown 时 ERROR 日志泄漏 |
