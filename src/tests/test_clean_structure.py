from pathlib import Path
import unittest


class CleanStructureTests(unittest.TestCase):
    def test_no_deleted_controller_references(self):
        root = Path(__file__).resolve().parents[2]
        forbidden = [
            "controller" + "_v2",
            "Stackelberg" + "V2",
            "run" + "_v2",
            "follower" + "_agents",
        ]
        for path in root.rglob("*"):
            if path.is_dir() or ".git" in path.parts or "outputs" in path.parts:
                continue
            if path.name == "test_clean_structure.py":
                continue
            if path.suffix not in {".py", ".md", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                self.assertNotIn(token, text, f"{token} found in {path}")


if __name__ == "__main__":
    unittest.main()
