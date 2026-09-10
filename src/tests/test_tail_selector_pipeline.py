from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rl_leader.run_tail_selector_pipeline import (
    DRAIN_OUT_FORMAT,
    discover_complete_drain_artifacts,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TailSelectorPipelineTests(unittest.TestCase):
    def test_discovery_uses_only_complete_passed_nonempty_drain_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root / "a" / "complete.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "status": "complete",
                    "passed": True,
                    "outcomes": [{"candidate_id": "a"}],
                },
            )
            _write(
                root / "b" / "running.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "status": "running",
                    "passed": False,
                    "outcomes": [],
                },
            )
            _write(
                root / "c" / "empty.json",
                {
                    "format_version": DRAIN_OUT_FORMAT,
                    "status": "complete",
                    "passed": True,
                    "outcomes": [],
                },
            )
            _write(
                root / "d" / "other.json",
                {
                    "format_version": "not_tail_drain",
                    "status": "complete",
                    "passed": True,
                    "outcomes": [{"candidate_id": "d"}],
                },
            )

            paths = discover_complete_drain_artifacts([root])

        self.assertEqual([path.name for path in paths], ["complete.json"])


if __name__ == "__main__":
    unittest.main()
