# Active Context — Project_hand

## 当前焦点

**CRP Phase D 逻辑修复 + Python 端协议对齐完成。待实机全流程测试。**

## 最近变更

1. **CRP Phase D 全面修复** (2026-07-14):
   - 新增变量: strflag_cont, strflag_abort (INT=1), str7 ("READY"), socketstr5
   - TranStrToCpos operateval 分离: as → strflag_cont / strflag_abort
   - If 条件修正: socketflag != 0 → strflag_cont == 0 / strflag_abort == 0 (operator: = → ==)
   - CONTINUE 分支: 新增 DCp2 + C_P4 计算 (DCp2.dz=Home.z-C_P1.z, CposOffset)
   - 标签规范化: Phase D widget 去括号后缀, 对齐 greeting CRP
   - Phase D Socket 引用修正: Socket1 → Socket0 (If 块内 3 处)

2. **Python RELEASE 三路分支修正** (2026-07-14):
   - abort_mode 提到优先判断 (ABORT 路径 RELEASE 不再误入 Phase C)
   - 人未就位不再提前发 ABORT, 等 RELEASE 到来走正常判定
   - hand_in_cup_active 保持 True 至 RELEASE 判定完成

3. **Cup 检测修复** (2026-07-14):
   - 补全 cup_detector: depth_intrinsics / color_to_depth_extrinsics / depth_to_color_extrinsics

4. **Camera B 优化** (2026-07-14):
   - YOLO device: gpu → cpu (与 Camera A 解耦)
   - 生命周期: 回到 GREETING enter 启动 (YOLO 在 Phase 0 预热)

5. **退出加速** (2026-07-14):
   - disable_hand(skip_reset=True) 跳过 RESET 手势
   - HandInCup.close() daemon 线程 + 2s 超时

## 未解决问题

- **实机双相机全流程测试待进行**

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **双相机全流程测试** | `Orchestrator/main.py` + `handlepluscup-orchestrator.crp` |
| 🟡 | Camera B 杯 ROI 标定 | UGREEN 替换后首次 |
