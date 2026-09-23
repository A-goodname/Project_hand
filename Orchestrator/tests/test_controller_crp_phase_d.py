"""Static checks for controller-export-derived PhaseD motion branches."""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "doc" / "controller_exports"


def main_contents(path):
    project = json.loads(path.read_text(encoding="utf-8"))
    task = next(x for x in project["children"] if x.get("typetask") == "main")
    widgets = next(x["children"] for x in task["children"] if x["type"] == "widgets")
    points = next(x["children"] for x in task["children"] if x["type"] == "points")
    return project, widgets, {p["id"]: p["label"] for p in points}


class ControllerPhaseDBranchTest(unittest.TestCase):
    def check_branches(self, filename):
        if not (EXPORTS / filename).is_file():
            self.skipTest(f"Local controller export is not available: {filename}")
        _, widgets, point_labels = main_contents(EXPORTS / filename)
        self.assertEqual([w["type"] for w in widgets[-4:]],
                         ["transtrtocpos", "transtrtocpos", "if", "if"])
        self.assertEqual([w["data"]["value"]["strstring"] for w in widgets[-4:-2]],
                         ["<", "["])
        for parser in widgets[-4:-2]:
            target = parser["data"]["value"]["pointa"]["pointvar"]["value"]
            self.assertEqual(point_labels[target], "C_P3")
        cont, abort = widgets[-2:]
        self.assertEqual([x["type"] for x in cont["children"]],
                         ["wait", "movl", "socketclose"])
        self.assertEqual(cont["children"][0]["data"]["value"]["time"], 1000)
        self.assertEqual(point_labels[cont["children"][1]["data"]["value"]["pointa"]["pointvar"]["value"]], "Home")
        self.assertEqual([x["type"] for x in abort["children"]],
                         ["movl", "socketsendstr", "wait", "movl",
                          "socketsendstr", "wait", "movl", "socketclose"])
        self.assertEqual([point_labels[abort["children"][i]["data"]["value"]["pointa"]["pointvar"]["value"]]
                          for i in (0, 3, 6)], ["C_P1", "C_P2", "Home"])
        self.assertEqual([abort["children"][i]["data"]["value"]["msendvar"]["datavar"]["value"]
                          for i in (1, 4)], ["str5", "str3"])
        self.assertEqual(abort["children"][0]["data"]["value"]["mspeed"]["datavar"]["value"], "sudu2")
        for branch in (cont, abort):
            for child in branch["children"]:
                self.assertEqual(child["parents"], [branch["id"]])

    def test_standalone_a2d(self):
        self.check_branches("PhaseA2D-KeyboardGate.crp")

    def test_fused_all_without_rps(self):
        self.check_branches("PhaseAll-NoRPS.crp")
        project, widgets, _ = main_contents(EXPORTS / "PhaseAll-NoRPS.crp")
        self.assertEqual(sum(w["type"] == "socketcreate" for w in widgets), 2)
        self.assertNotIn("8700", json.dumps(project))


if __name__ == "__main__":
    unittest.main()
