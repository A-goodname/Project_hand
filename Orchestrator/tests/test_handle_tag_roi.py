# -*- coding: utf-8 -*-
"""青色贴纸搜索区域扩展 (2026-08-20).

覆盖 _detect_tag_region 的百分比 ROI:
  1. 右侧把手 [70%, 110%] 区域 → 拾取区域外延贴纸
  2. 左侧把手 [-10%, 30%] 区域 → 拾取左侧外延贴纸
  3. 区域外贴纸不被拾取
  4. ROI clamp 到图像边界 (bbox 贴右缘不越界)
  5. 空区域 (rx1<=rx0) → None
"""
import unittest

import numpy as np
import cv2  # noqa: F401  (ensure cv2 import path works)

from Orchestrator.handle_cup_detector import _detect_tag_region


def _green_img(shape=(200, 200)):
    """黑背景 (V=0 排除) 图像."""
    return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


def _put_tag(img, x0, x1, y0, y1):
    """在 [x0:x1, y0:y1] 画纯绿矩形 (HSV H=60/S=255/V=255, 匹配阈值)."""
    img[y0:y1, x0:x1] = (0, 255, 0)


class HandleTagRoiTest(unittest.TestCase):
    def test_right_region_picks_extended_tag(self):
        """右侧 [70%,110%]: bbox(50,50,150,150) w=100 → ROI x∈[120,160];
        贴纸落在 130..155 (含旧 70%~100% 之外的右侧外延) 应被拾取."""
        img = _green_img()
        _put_tag(img, 130, 155, 60, 140)
        res = _detect_tag_region(img, (50, 50, 150, 150), "right")
        self.assertIsNotNone(res)
        u, v, area, angle, a1, a2, mask = res
        self.assertGreaterEqual(u, 120)
        self.assertLessEqual(u, 160)
        self.assertGreater(area, 0)

    def test_left_region_picks_extended_tag(self):
        """左侧 [-10%,30%]: bbox(50,50,150,150) w=100 → ROI x∈[40,80];
        贴纸落在 45..75 (含向左外延) 应被拾取."""
        img = _green_img()
        _put_tag(img, 45, 75, 60, 140)
        res = _detect_tag_region(img, (50, 50, 150, 150), "left")
        self.assertIsNotNone(res)
        u, v, area, angle, a1, a2, mask = res
        self.assertGreaterEqual(u, 40)
        self.assertLessEqual(u, 80)

    def test_tag_outside_region_not_picked(self):
        """贴纸在右侧 ROI [120,160] 之外 (100..115) → 不拾取."""
        img = _green_img()
        _put_tag(img, 100, 115, 60, 140)
        self.assertIsNone(_detect_tag_region(img, (50, 50, 150, 150), "right"))

    def test_region_clamped_right_edge(self):
        """bbox 贴图像右缘 (x1=195, w=15): 1.1w 外延超图像 → rx1 clamp 到 200,
        不越界不抛异常; clamp 后区域内的贴纸可拾取."""
        img = _green_img()
        # 无贴纸: 不越界即 OK
        self.assertIsNone(_detect_tag_region(img, (180, 50, 195, 150), "right"))
        img2 = _green_img()
        _put_tag(img2, 193, 198, 60, 140)
        res = _detect_tag_region(img2, (180, 50, 195, 150), "right")
        self.assertIsNotNone(res)
        u, v, area, angle, a1, a2, mask = res
        self.assertLessEqual(u, 200)

    def test_empty_region_returns_none(self):
        """空区域 (rx1<=rx0): bbox x0=200 贴右缘, 右侧区间全被 clamp 出界."""
        img = _green_img()
        self.assertIsNone(_detect_tag_region(img, (200, 50, 201, 150), "right"))

    def test_default_handle_side_is_right(self):
        """handle_side 缺省 = right (向后兼容)."""
        img = _green_img()
        _put_tag(img, 130, 155, 60, 140)
        self.assertIsNotNone(_detect_tag_region(img, (50, 50, 150, 150)))


if __name__ == "__main__":
    unittest.main()
