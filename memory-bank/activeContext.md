# Active Context — Project_hand

## 当前焦点

**语音手势控制完善** — 单手势测试通过，序列+速度待测；ASR 精度优化中。

## 最近变更

### voice_omnihand_control.py 完善 (2026-07-17)

1. **三档速度前缀齐全**: 无前缀=直接(最快), 快速/中速/慢速=插值
2. **预定义序列**: 打招呼(HELLO×3), 握手(压力检测), 数字手势(1→8→合十)
3. **停止指令全局生效**: "停止/急停/暂停/停下/别动/快停" → 中断序列 + RESET
4. **防误触**: 序列匹配先于手势(避免"握手序列"被"握手"截获)；关键词占比检查(>6字且<30%跳过)；删"吧""好的""可以"等高频词
5. **mic可选**: `--mic auto/ugreen/default`，UGREEN 优先查找
6. **模型可选**: `--model auto/full/int8`，完整版 model.onnx 优先
7. **AutoVAD**: `q`键退出(msvcrt)，静音阈值 1.2s
8. **silero_vad.onnx**: 尝试接入失败(`is_speech`无状态，CFFI回调崩溃)，已回退能量阈值，模型保留在 model/

### 主流程优化 v15 (2026-07-17)
- P3超时延长、手动CONTINUE按键、人就位简化(纯人检测)、Camera B自动探测、关键词防误触

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | ASR 精度优化 | 换完整 model.onnx (895MB) 替代 int8 (227MB) |
| 🔴 | 默认 mic 测试 | `--mic default` 对比 UGREEN mic |
| 🟡 | voice_omnihand_control 序列+速度 实机测 | 连灵巧手测试 |
| 🟡 | HandInCup 继续优化 | landmarks-in-bbox 调参，遮挡场景 |
| 🟡 | 加水实测 | 完整流程加水倒水 |
| 🟢 | 完整语音控制接入 Orchestrator | voice_control.py + CRP GOTO |
| 🟢 | MediaPipe clearcut 彻底抑制 | 