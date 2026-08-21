# -*- coding: utf-8 -*-
"""Phase 0 语音欢迎接入 — 检测到人进门时异步触发语音播报 (无相机/灵巧手/音箱).

覆盖 Estun/estun_greeting_phase0.py 的 _announce_voice + _wait_for_person 触发逻辑:
  1. 检测到人进门 → 调 _announce_voice, Popen 参数为 bash ... --no-enter (**不传 --message,
     文案全部复用 /data/Speech config.json greeting.message, 我方零硬编码)
  2. voice.enabled=false → 不播报
  3. 语音脚本不存在 → 安全跳过 (不抛异常)
  4. _wait_for_person 检测到人时返回 True (mock frame + mock _person_detected)
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from Estun.estun_greeting_phase0 import Phase0Greeting  # noqa: E402


def _make_cfg(voice_enabled=True,
              voice_script="/data/Speech/start_greeting.sh"):
    return {
        "greeting": {
            "zhaoshou1_pose": [0.0] * 6,
            "zhaoshou2_pose": [0.0] * 6,
            "woshou_pose": [0.0] * 6,
            "wave_cycles": 1,
            "wave_interval": 0.05,
            "force_timeout": 0.1,
            "force_release_timeout": 0.1,
            "force_poll_interval": 0.01,
            "force_release_hold": 0.02,
            "voice": {
                "enabled": voice_enabled,
                "script": voice_script,
            },
        },
    }


class Phase0VoiceTest(unittest.TestCase):

    def _svc(self, **kwargs):
        svc = Phase0Greeting(_make_cfg(**kwargs), camera_path="fake",
                             preview=False)
        svc._hand = None   # 无灵巧手, 模拟
        svc._camera = None
        return svc

    # ---------- 1. 检测到人进门 → 触发播报 (复用对方文案, 不传 --message) ----------

    def test_person_detected_triggers_voice(self):
        svc = self._svc()
        # mock 掉依赖: _person_detected 恒 True → 一次循环即 stable 满
        with mock.patch.object(svc, "_person_detected", return_value=True), \
             mock.patch.object(svc, "_get_frame", return_value="frame"), \
             mock.patch("subprocess.Popen") as popen:
            ok = svc._wait_for_person(timeout=1.0)
        self.assertTrue(ok, "检测到人应返回 True")
        popen.assert_called_once()
        args = popen.call_args[0][0]
        self.assertEqual(args[0], "bash")
        self.assertIn("--no-enter", args)
        self.assertNotIn("--message", args,
                         "文案不自定义: 复用 /data/Speech config.json greeting.message")

    def test_person_timeout_no_voice(self):
        svc = self._svc()
        with mock.patch.object(svc, "_person_detected", return_value=False), \
             mock.patch.object(svc, "_get_frame", return_value="frame"), \
             mock.patch("subprocess.Popen") as popen:
            ok = svc._wait_for_person(timeout=0.3)
        self.assertFalse(ok, "超时不应返回 True")
        popen.assert_not_called()

    # ---------- 2. enabled=false 不播报 ----------

    def test_voice_disabled_no_announce(self):
        svc = self._svc(voice_enabled=False)
        with mock.patch.object(svc, "_person_detected", return_value=True), \
             mock.patch.object(svc, "_get_frame", return_value="frame"), \
             mock.patch("subprocess.Popen") as popen:
            ok = svc._wait_for_person(timeout=1.0)
        self.assertTrue(ok, "关语音仍应检测人")
        popen.assert_not_called()

    # ---------- 3. 脚本不存在 → 安全跳过 ----------

    def test_missing_script_skips_gracefully(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "no_such.sh")
            svc = self._svc(voice_script=missing)
            with mock.patch("subprocess.Popen") as popen:
                svc._announce_voice()
            popen.assert_not_called()

    # ---------- 4. 直接调 _announce_voice: Popen 非阻塞 ----------

    def test_announce_voice_spawns_and_returns(self):
        svc = self._svc()
        with mock.patch("subprocess.Popen") as popen:
            svc._announce_voice()
        popen.assert_called_once()
        call = popen.call_args
        self.assertEqual(call.kwargs["start_new_session"], True)

    # ---------- 5. stop() 显式关闭灵巧手 (review B 点) ----------

    def test_stop_closes_hand(self):
        svc = self._svc()
        svc._hand = object()   # 模拟有灵巧手
        with mock.patch("Estun.utils_shared.disable_hand") as dh:
            svc.stop()
        dh.assert_called_once()
        self.assertIsNone(svc._hand, "stop 后应置空 _hand")

    def test_stop_hand_close_failure_safe(self):
        svc = self._svc()
        svc._hand = object()
        with mock.patch("Estun.utils_shared.disable_hand",
                        side_effect=RuntimeError("串口忙")):
            svc.stop()   # 不抛异常
        self.assertIsNone(svc._hand)


if __name__ == "__main__":
    unittest.main()
