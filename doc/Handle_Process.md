# 把手检测流程: YOLO-World + SAM + 水平扫描线

## 流程概览

```
彩色图
  │
  ├─→ YOLO-World (GPU) → 水壶 bbox
  │
  ├─→ SAM (CPU) → 水壶完整 mask (壶身+把手), 倾斜放置更容易获得良好mask，对位置要求很高
  │
  ├─→ 水平扫描线分离把手
  │    └─ 每行: mask|gap|mask 模式 → 小段 = 把手
  │
  ├─→ 后处理
  │    ├─ MORPH_CLOSE 填充把手内部空洞
  │    ├─ 只保留最大连通分量 (排除噪声)
  │    ├─ 面积比+质心校验
  │    └─ X 中心一致性过滤: 从底部向上剔除 x 中心偏离过大的行
  │         (解决 C 形把手底座被误检为把手的问题)
  │
  └─→ 输出: handle_mask + handle_bbox
```

## 核心原理

SAM 生成的水壶 mask 是一个完整的连通域 (壶身+把手连在一起). 当把手与壶身之间存在间隙时, 每一行从左到右扫描会呈现 `mask|gap|mask` 模式:

- 第一个 mask 段 = 壶身
- gap (无 mask) = 把手与壶身之间的空隙
- 第二个 mask 段 = 把手

取每行两个 mask 段中 **面积较小者** 标记为把手, 所有行的把手区域取并集.

## 校验规则

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `handle_side` | `"right"` | 把手在水壶左侧还是右侧 |
| `handle_min_ratio` | `0.05` (5%) | 把手面积 / 壶身面积 下限 |
| `handle_max_ratio` | `0.55` (55%) | 把手面积 / 壶身面积 上限 |
| `erode_max_iter` | `15` | 可选腐蚀迭代次数 (保留) |
| `handle_center_deviation_ratio` | `0.3` (30%) | X 中心偏差容差(相对把手宽度比例) |

此外附加:
- **质心校验**: 把手质心必须在水壶 bbox 中心指定侧 (左/右)
- **最小行数**: 至少 3 行检测到 mask|gap|mask 模式
- **最小面积**: 把手至少 20 像素
- **X 中心一致性**: 计算各行的把手 x 中心, 用中位数 + `handle_center_deviation_ratio` × 平均宽度 作为容差, 从底部向上剔除偏差过大的行 (解决 C 形把手底座误检)

## 局限性 (当前已知)

1. **把手必须在壶身 mask 内部**: 若把手与壶身间隙过大, YOLO-World bbox 只框住壶身, 把手可能部分在框外导致 SAM 未分割到 → 当前无两阶段扩展搜索兜底
2. **壶身+把手必须是单个连通域**: 若 SAM 将壶身和把手分割为两个独立 mask, 水平扫描线法不适用
3. **只支持单侧把手**: 通过 `handle_side` 指定 "right" 或 "left"
4. **把手不能太厚**: 若把手宽度接近壶身宽度, 两个 mask 段面积相近, 无法区分
5. **X 中心一致性过滤影响稳定性**: 当把手倾斜较大时, x 中心沿把手自然变化可能超出阈值导致误切; 不同帧间 SAM mask 质量变化导致统计量不稳定, 可能引起 mask 波动

## 关键文件

| 文件 | 说明 |
|------|------|
| `RealSense_R/test_SAM_handle.py` | 主程序 (RealSenseHandleDetector 类) |
| `RealSense_R/config.yaml` | 配置文件 (handle 配置段) |
| `RealSense_R/D435i_info.yaml` | RealSense D435i 相机参数 |
| `RealSense_R/realsense_shared.py` | 共享模块 (`scanline_find_handle` 函数) |

## 运行方法

```bash
D:\Environment\miniconda\envs\Proj\python.exe RealSense_R/test_SAM_handle.py
```

- `q` 键退出
- `s` 键保存当前帧 (原图+mask) 至 `RealSense_R/images/`

## 可视化

| 颜色 | 含义 |
|------|------|
| 蓝色半透明 | 壶身 mask |
| 绿色半透明 + 绿色轮廓 | 把手 mask |
| 绿色矩形 | 把手 axis-aligned bbox |
| 黄色旋转矩形 | 把手 `minAreaRect` 旋转矩形 (更贴合) |