# -*- coding: utf-8 -*-
"""现场门禁 site_gate 编排测试 (audit 修复 1, 2026-08-16).

硬件检查函数经 monkeypatch 注入, 验证编排逻辑:
  失败聚合 / 现场准备确认 / 网线链路 警告 vs 阻断.
"""
import unittest

import Orchestrator.site_gate as sg


def _pass_checks():
    sg.check_camera_a_serial = lambda s: None
    sg.check_camera_b_physical_port = lambda p: None
    sg.check_hand_carrier_alive = lambda p: None
    sg.check_ether_link = lambda i: (None, "UP")


class SiteGateTest(unittest.TestCase):
    def test_preparation_and_camera_failures(self):
        """现场准备未确认 + Camera A 序列号不匹配 → 两项失败."""
        _pass_checks()
        sg.check_camera_a_serial = lambda s: "序列号不匹配: 期望 X, 实际 []"
        fails = sg.run_site_gate(preparation_ok=False, camera_a_serial="X",
                                 camera_b_path="p", hand_serial_path="h")
        self.assertGreaterEqual(len(fails), 2)
        self.assertTrue(any("准备" in f for f in fails))
        self.assertTrue(any("序列号" in f for f in fails))

    def test_camera_b_and_hand_failures(self):
        """Camera B 物理口 + 灵巧手串口缺失 → 两项失败."""
        _pass_checks()
        sg.check_camera_b_physical_port = lambda p: "Camera B 物理口不匹配"
        sg.check_hand_carrier_alive = lambda p: "灵巧手串口不存在"
        fails = sg.run_site_gate(preparation_ok=True, camera_a_serial="X",
                                 camera_b_path="p", hand_serial_path="h")
        self.assertEqual(len(fails), 2)
        self.assertTrue(any("物理口" in f for f in fails))
        self.assertTrue(any("串口" in f for f in fails))

    def test_all_pass(self):
        """全部通过 → 无失败."""
        _pass_checks()
        fails = sg.run_site_gate(preparation_ok=True, camera_a_serial="X",
                                 camera_b_path="p", hand_serial_path="h")
        self.assertEqual(fails, [])

    def test_ether_link_warn_default(self):
        """网线 NO-CARRIER 默认仅警告, 不阻断."""
        _pass_checks()
        sg.check_ether_link = lambda i: ("NO-CARRIER 需核验", "NO-CARRIER")
        fails = sg.run_site_gate(preparation_ok=True, camera_a_serial="X",
                                 camera_b_path="p", hand_serial_path="h")
        self.assertEqual(fails, [])

    def test_ether_link_block(self):
        """block_link_down=true → 网线断开阻断启动."""
        _pass_checks()
        sg.check_ether_link = lambda i: ("NO-CARRIER 需核验", "NO-CARRIER")
        fails = sg.run_site_gate(preparation_ok=True, camera_a_serial="X",
                                 camera_b_path="p", hand_serial_path="h",
                                 block_link_down=True)
        self.assertEqual(len(fails), 1)
        self.assertIn("NO-CARRIER", fails[0])


if __name__ == "__main__":
    unittest.main()
