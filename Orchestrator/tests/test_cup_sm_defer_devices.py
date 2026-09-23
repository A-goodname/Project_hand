# -*- coding: utf-8 -*-
"""A2D estun_handle_cup_sm.py --defer-devices 融合模式测试 (无硬件, mock 设备 init).

覆盖 (三 CRP 融合 2026-08-21; 方案甲 2026-08-22 Camera B 交 merged 进程, 本状态机恒不初始化):
  1. defer 模式 start(): 只 init 配置/标定/稳定器/TCP, 不 init 灵巧手/D435i
  2. 非 defer 模式 start(): 启灵巧手/D435i (Camera B 仍不初始化, 已交 merged)
  3. _ensure_devices(): 首次连接开设备, 二次跳过 (幂等)
  4. _ensure_devices(): 非 defer 模式 no-op
"""
import os
import sys
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from Orchestrator.estun_handle_cup_sm import HandleCupSM  # noqa: E402


class CupSmDeferDevicesTest(unittest.TestCase):

    def _sm(self, defer):
        sm = HandleCupSM()
        sm._defer_devices = defer
        return sm

    def test_defer_start_skips_devices(self):
        """defer 模式 start(): 只 init 配置/标定/稳定器/TCP, 不 init 设备."""
        sm = self._sm(defer=True)
        with mock.patch.object(sm, "_load_configs") as m_cfg, \
             mock.patch.object(sm, "_init_stabilizer_and_calib") as m_stab, \
             mock.patch.object(sm, "_init_tcp") as m_tcp, \
             mock.patch.object(sm, "_init_hand") as m_hand, \
             mock.patch.object(sm, "_init_detector") as m_det:
            sm.start()
        m_cfg.assert_called_once()
        m_stab.assert_called_once()
        m_tcp.assert_called_once()
        m_hand.assert_not_called()
        m_det.assert_not_called()

    def test_normal_start_opens_hand_and_det(self):
        """非 defer 模式 start(): 启灵巧手/D435i (Camera B 已交 merged, 不初始化)."""
        sm = self._sm(defer=False)
        with mock.patch.object(sm, "_load_configs"), \
             mock.patch.object(sm, "_init_stabilizer_and_calib") as m_stab, \
             mock.patch.object(sm, "_init_tcp") as m_tcp, \
             mock.patch.object(sm, "_init_hand") as m_hand, \
             mock.patch.object(sm, "_init_detector") as m_det, \
             mock.patch.object(sm, "_prepare_handover_vision") as m_hic:
            sm.start()
        m_hand.assert_called_once()
        m_det.assert_called_once()
        m_stab.assert_called_once()
        m_tcp.assert_called_once()
        m_hic.assert_called_once()

    def test_ensure_devices_idempotent(self):
        """_ensure_devices: 首次打开设备, 二次跳过 (幂等)."""
        sm = self._sm(defer=True)
        with mock.patch.object(sm, "_init_hand") as m_hand, \
             mock.patch.object(sm, "_init_detector") as m_det:
            sm._ensure_devices()
            sm._ensure_devices()
        m_hand.assert_called_once()
        m_det.assert_called_once()

    def test_ensure_devices_noop_when_not_defer(self):
        """非 defer 模式: _ensure_devices 不动作 (设备已在 start 打开)."""
        sm = self._sm(defer=False)
        with mock.patch.object(sm, "_init_hand") as m_hand, \
             mock.patch.object(sm, "_init_detector") as m_det:
            sm._ensure_devices()
        m_hand.assert_not_called()
        m_det.assert_not_called()

    def test_run_accepts_tcp_when_no_frames(self):
        """defer 模式: 检测器未开(无帧)时主循环仍 accept — 修复 8080 永不接受连接 bug.

        原 run(): _accept_tcp 只在帧非 None 分支调用; defer 下 _get_frames 恒 None → continue
        → 永不 accept → CRP 连 8080 无响应, 设备也永远不打开 (2026-08-21 实机定位).
        """
        sm = self._sm(defer=True)
        with mock.patch("cv2.namedWindow"), \
             mock.patch("cv2.waitKey", return_value=255), \
             mock.patch("cv2.destroyAllWindows"), \
             mock.patch.object(sm, "_accept_tcp",
                               side_effect=lambda: sm._stop_event.set()) as m_accept:
            sm.run()
        m_accept.assert_called()
        self.assertTrue(sm._devices_opened is False)  # 仅验证 accept 被调用, 设备仍待连接时开

    def test_no_preview_skips_windows(self):
        """--no-preview: run() 不建 cv2 窗口 (headless; 融合演示多窗口 CPU 渲染太卡)."""
        sm = self._sm(defer=True)
        sm._no_preview = True
        with mock.patch("cv2.namedWindow") as m_win, \
             mock.patch.object(sm, "_accept_tcp",
                               side_effect=lambda: sm._stop_event.set()) as m_accept:
            sm.run()
        m_win.assert_not_called()
        m_accept.assert_called()


if __name__ == "__main__":
    unittest.main()
