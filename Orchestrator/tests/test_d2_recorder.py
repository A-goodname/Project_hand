# -*- coding: utf-8 -*-
"""D2 记录器 + data_sm 接收端改造测试 (2026-08-17).

覆盖:
  1. 前缀映射: P0/P1/P2/P3/P4/Home 按出现次序 → D2 九事件 (P1/P2/P4 二次映射)
  2. 次数越界 / 未知前缀 / 字段错误 / NaN / 单位错误 → 拒绝
  3. EpisodeRecorder: robot_state.jsonl schema + 双 manifest + 顺序判定
  4. 分号帧 + 裸命令混合分发 (parse 层)
"""
import json
import os
import tempfile
import unittest

import numpy as np

from Orchestrator.d2_recorder import (
    EpisodeRecorder, PREFIX_EVENT_MAP, parse_pose_prefix,
)
from Orchestrator.d2_streams import WristStreamRecorder  # noqa: E402
from Orchestrator.estun_handle_data_sm import HandleDataSM   # noqa: E402

P = (0.1, 0.9, 0.3)
A = (-115.2, 85.0, -21.1)


def _f(prefix, xyz=P, abc=A):
    """现行 CRP 格式: <PREFIX>[x,y,z,a,b,c] (CPosToStr strstring='P2[' endstring=']')."""
    return (f"{prefix}[{xyz[0]},{xyz[1]},{xyz[2]},"
            f"{abc[0]},{abc[1]},{abc[2]}]")




class PrefixMapTest(unittest.TestCase):
    def test_full_sequence_mapping(self):
        """按 D2 §3.2 顺序发送前缀 → 映射为完整九事件."""
        occ = {}
        got = []
        for prefix in ["P0", "P2", "P1", "P4", "P3", "P4", "P1", "P2", "Home"]:
            event, xyz, abc = parse_pose_prefix(_f(prefix), occ)
            got.append(event)
        self.assertEqual(got, [
            "P0_ACTUAL", "P2_ARRIVED", "P1_ARRIVED", "P4_AFTER_LIFT",
            "P3_ARRIVED", "P4_AFTER_POUR", "P1_RETURNED",
            "P2_FINAL_ARRIVED", "HOME_ARRIVED",
        ])

    def test_occurrence_overflow(self):
        """P4 第 3 次 / P1 第 3 次 → 拒绝 (超出映射)."""
        occ = {}
        parse_pose_prefix(_f("P4"), occ)
        parse_pose_prefix(_f("P4"), occ)
        with self.assertRaises(ValueError):
            parse_pose_prefix(_f("P4"), occ)
        occ2 = {}
        parse_pose_prefix(_f("P1"), occ2)
        parse_pose_prefix(_f("P1"), occ2)
        with self.assertRaises(ValueError):
            parse_pose_prefix(_f("P1"), occ2)

    def test_unknown_prefix(self):
        with self.assertRaises(ValueError):
            parse_pose_prefix(_f("PX"), {})

    def test_bad_values(self):
        with self.assertRaises(ValueError):
            parse_pose_prefix("P2[0.1,0.2,0.3,1,2]", {})          # 5 字段
        with self.assertRaises(ValueError):
            parse_pose_prefix("P2[0.1,abc,0.3,1,2,3]", {})         # 非数值
        with self.assertRaises(ValueError):
            parse_pose_prefix("P2[nan,0.2,0.3,1,2,3]", {})         # NaN
        with self.assertRaises(ValueError):
            parse_pose_prefix("P2[111.9,378.3,698.3,1,2,3]", {})   # mm 误作 m
        with self.assertRaises(ValueError):
            parse_pose_prefix("0.1,0.2,0.3,1,2,3", {})             # 无前缀

    def test_map_structure(self):
        self.assertEqual(PREFIX_EVENT_MAP["P0"], ("P0_ACTUAL",))
        self.assertEqual(PREFIX_EVENT_MAP["Home"], ("HOME_ARRIVED",))
        self.assertEqual(len(PREFIX_EVENT_MAP), 6)


class EpisodeRecorderTest(unittest.TestCase):
    def test_records_and_manifests(self):
        out = tempfile.mkdtemp(prefix="d2rec-")
        rec = EpisodeRecorder(out, "ep-rec-001", "canonical-a")
        for prefix in ["P0", "P2", "P1", "P4", "P3", "P4", "P1", "P2", "Home"]:
            event, xyz, abc = parse_pose_prefix(_f(prefix), rec._occurrences)
            rec.record_pose(event, xyz, abc, raw_payload=_f(prefix))
        rec.record_receipt("gripdone", command="GRIPDONE")
        rec.close()
        # robot_state.jsonl
        with open(os.path.join(out, "ep-rec-001", "signals",
                               "robot_state.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertEqual([r["event"] for r in rows], [
            "P0_ACTUAL", "P2_ARRIVED", "P1_ARRIVED", "P4_AFTER_LIFT",
            "P3_ARRIVED", "P4_AFTER_POUR", "P1_RETURNED",
            "P2_FINAL_ARRIVED", "HOME_ARRIVED"])
        self.assertEqual(rows[0]["schema_version"], "estun-cpos-event-v1")
        self.assertEqual(rows[0]["state_kind"], "ESTUN_TCP_EVENT_SNAPSHOT")
        # episode_manifest
        with open(os.path.join(out, "ep-rec-001",
                               "episode_manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        self.assertTrue(man["sequence_complete"])
        self.assertEqual(man["outcome"], "SUCCESS")
        self.assertFalse(man["expert_valid"])    # 保守待审计
        # artifact_manifest
        with open(os.path.join(out, "ep-rec-001",
                               "artifact_manifest.json"), encoding="utf-8") as f:
            art = json.load(f)
        # robot_state + state_transition + action_receipt + force_torque(UNAVAILABLE)
        self.assertEqual(len(art["files"]), 4)
        self.assertTrue(all(fi["sha256"] for fi in art["files"]))
        # 回执
        with open(os.path.join(out, "ep-rec-001", "signals",
                               "action_receipt.jsonl"), encoding="utf-8") as f:
            receipts = [json.loads(l) for l in f if l.strip()]
        self.assertTrue(any(r.get("command") == "GRIPDONE"
                            for r in receipts))

    def test_invalid_marks_episode(self):
        """解析失败 → invalid_reasons 记录, 顺序不完整 → ABORTED."""
        out = tempfile.mkdtemp(prefix="d2rec-")
        rec = EpisodeRecorder(out, "ep-rec-002", "canonical-a")
        rec.record_pose("P0_ACTUAL", P, A, raw_payload=_f("P0"))
        rec.note_invalid("位姿帧解析失败: 未知前缀 'PX'")
        rec.close()
        with open(os.path.join(out, "ep-rec-002",
                               "episode_manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        self.assertFalse(man["sequence_complete"])
        self.assertEqual(man["outcome"], "ABORTED")
        self.assertTrue(any("解析失败" in r for r in man["invalid_reasons"]))


class StreamRecorderTest(unittest.TestCase):
    """raw/ 流记录: 腕部合成帧 + hand/force JSONL + manifest 含 raw."""

    def test_wrist_stream_synthetic(self):
        out = tempfile.mkdtemp(prefix="d2stream-")
        wr = WristStreamRecorder(out, fps=10.0)
        wr.start()
        for i in range(10):
            color = np.full((480, 640, 3), i * 10, dtype=np.uint8)
            depth = np.full((480, 640), i, dtype=np.uint16)
            wr.record_frame(color, depth, i)
        wr.close()
        raw = os.path.join(out, "raw")
        self.assertTrue(os.path.exists(os.path.join(raw, "wrist_rgb.mp4")))
        self.assertTrue(os.path.exists(os.path.join(raw, "wrist_depth_000000.png")))
        self.assertTrue(os.path.exists(os.path.join(raw, "wrist_depth_000009.png")))
        with open(os.path.join(raw, "wrist_rgb_timestamps.jsonl"),
                  encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(rows), 10)
        with open(os.path.join(raw, "wrist_depth_meta.json"),
                  encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["depth_dtype"], "uint16")
        self.assertEqual(meta["frames"], 10)

    def test_hand_and_force_records(self):
        """hand_state (命令事件) + force_torque (UNAVAILABLE) 落盘."""
        out = tempfile.mkdtemp(prefix="d2stream-")
        rec = EpisodeRecorder(out, "ep-stream-001", "canonical-a")
        rec.record_hand(command="GRIPDONE", hand_available=True)
        rec.close()
        with open(os.path.join(out, "ep-stream-001", "signals",
                               "hand_state.jsonl"), encoding="utf-8") as f:
            hands = [json.loads(l) for l in f if l.strip()]
        self.assertTrue(any(h.get("command") == "GRIPDONE" for h in hands))
        with open(os.path.join(out, "ep-stream-001", "signals",
                               "force_torque.jsonl"), encoding="utf-8") as f:
            forces = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(forces), 1)
        self.assertEqual(forces[0]["health"], "UNAVAILABLE")
        self.assertIsNone(forces[0]["values"])     # 禁填零伪造
        # manifest 覆盖 signals 全部工件
        with open(os.path.join(out, "ep-stream-001",
                               "artifact_manifest.json"), encoding="utf-8") as f:
            art = json.load(f)
        art_names = {fi["relpath"] for fi in art["files"]}
        self.assertIn("signals/hand_state.jsonl", art_names)
        self.assertIn("signals/force_torque.jsonl", art_names)
        # python_authorization_enabled=false 声明
        with open(os.path.join(out, "ep-stream-001",
                               "episode_manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        self.assertIs(man["python_authorization_enabled"], False)


class DataSmDispatchTest(unittest.TestCase):
    """data_sm _dispatch 集成: 位姿事件 + 命令混合分发 (免硬件)."""

    def test_dispatch_pose_and_commands(self):
        sm = HandleDataSM()          # __init__ 轻量, 无硬件 (start 才初始化)
        out = tempfile.mkdtemp(prefix="d2sm-")
        rec = EpisodeRecorder(out, "ep-sm-001", "canonical-a")
        occ = {}
        lines = [
            "P0[0.1,0.9,0.3,-115,85,-21]",
            "P2[0.13,1.11,0.295,-125.6,0.7,4.8]",
            "PREGRIP",
            "P1[0.06,0.985,0.3,-115.2,85,-21.1]",
            "GRIPDONE",
            "P4[0.05,1.05,0.5,-115.2,85,-21.1]",
            "P3[0.07,1.02,0.42,-93.3,30.5,-5.3]",
            "P4[0.05,1.05,0.5,-115.2,85,-21.1]",
            "P1[0.06,0.985,0.3,-115.2,85,-21.1]",
            "P2[0.13,1.11,0.295,-125.6,0.7,4.8]",
            "Home[0.1119,0.3783,0.6983,-125.6,0.76,4.8]",
            "RESET",
        ]
        for line in lines:
            sm._dispatch(rec, occ, line.strip())
        rec.close()
        with open(os.path.join(out, "ep-sm-001", "signals",
                               "robot_state.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertEqual([r["event"] for r in rows], [
            "P0_ACTUAL", "P2_ARRIVED", "P1_ARRIVED", "P4_AFTER_LIFT",
            "P3_ARRIVED", "P4_AFTER_POUR", "P1_RETURNED",
            "P2_FINAL_ARRIVED", "HOME_ARRIVED"])
        # 命令 → 回执
        with open(os.path.join(out, "ep-sm-001", "signals",
                               "action_receipt.jsonl"), encoding="utf-8") as f:
            receipts = [json.loads(l) for l in f if l.strip()]
        commands = {r.get("command") for r in receipts if r.get("command")}
        self.assertEqual(commands, {"PREGRIP", "GRIPDONE", "RESET"})
        # 顺序完整
        with open(os.path.join(out, "ep-sm-001",
                               "episode_manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        self.assertTrue(man["sequence_complete"])


if __name__ == "__main__":
    unittest.main()
