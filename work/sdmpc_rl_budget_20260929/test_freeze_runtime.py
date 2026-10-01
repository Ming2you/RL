"""Standard-library tests; no original runtime imports or simulation calls."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import freeze_runtime as freeze


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="budget-snapshot-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.destination = self.root / "snapshot"
        self.raw = b'\xef\xbb\xbf{\r\n  "original": "C:\\\\original\\\\input.json"\r\n}\r\n'
        (self.source / "config.json").write_bytes(self.raw)
        (self.source / "code.py").write_bytes(b"# original CRLF\r\nVALUE = 42\r\n")
        self.selected = {"config.json": "fixture", "code.py": "runtime_source"}
        self.provenance = {
            "revision": "a" * 40, "branch": "original", "dirty": True,
            "status_sha256": "b" * 64, "status_porcelain_v1": "?? config.json\n",
            "warnings": [],
        }
        mocked = patch.object(freeze, "git_provenance", return_value=self.provenance)
        self.git = mocked.start()
        self.addCleanup(mocked.stop)

    def create_snapshot(self):
        return freeze.freeze_snapshot(self.source, self.destination, self.selected)

    def test_portable_relative_names(self):
        self.assertEqual(freeze.relative_path("work/a/config.json"), "work/a/config.json")
        invalid = ("", ".", "..", "../escape", "a/../b", "a/./b", "a//b", "/root",
                   "C:/root", "C:relative", "//server/share", "a\\b", "a/file:stream",
                   "a/CON.json", "a/file.", "a/file ", "a/fi?le", "a/\x00file")
        for name in invalid:
            with self.subTest(name=name), self.assertRaises(freeze.SnapshotError):
                freeze.relative_path(name)

    def test_disjoint_roots_required(self):
        for destination in (self.source, self.source / "nested", self.root):
            with self.subTest(destination=destination), self.assertRaises(freeze.SnapshotError):
                freeze.validate_roots(self.source, destination)
        with self.assertRaises(freeze.SnapshotError):
            freeze.validate_roots(self.root / "missing", self.destination)

    def test_missing_source_and_empty_selection_refused(self):
        for selected in ({}, {"missing.py": "runtime_source"}):
            with self.subTest(selected=selected), self.assertRaises(freeze.SnapshotError):
                freeze.freeze_snapshot(self.source, self.destination, selected)
        self.assertFalse(self.destination.exists())

    def test_case_collision_refused_before_writing(self):
        with self.assertRaises(freeze.SnapshotError):
            freeze.freeze_snapshot(self.source, self.destination, {"code.py": "source", "CODE.py": "source"})
        self.assertFalse(self.destination.exists())

    def test_symlink_or_junction_attributes_refused(self):
        original = Path.lstat
        linked = self.source / "config.json"

        def marked(path, *args, **kwargs):
            if path == linked:
                class Reparse:
                    st_mode = 0o100644
                    st_file_attributes = 0x400
                return Reparse()
            return original(path, *args, **kwargs)

        with patch.object(Path, "lstat", marked), self.assertRaises(freeze.SnapshotError):
            self.create_snapshot()
        self.assertFalse(self.destination.exists())

    def test_byte_hashes_provenance_and_offline_verification(self):
        source_before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.source.iterdir()}
        manifest = self.create_snapshot()
        self.assertEqual(manifest["file_count"], 2)
        self.assertEqual(manifest["source_git_before"], self.provenance)
        self.assertTrue(manifest["source_git_unchanged"])
        self.assertFalse(manifest["runtime_notes"]["historical_result_valid_for_new_timing_or_acceptance"])
        self.assertEqual((self.destination / "source/config.json").read_bytes(), self.raw)
        for entry in manifest["files"]:
            raw, mtime = source_before[entry["source_relative"]]
            digest = hashlib.sha256(raw).hexdigest()
            for key in ("sha256", "source_sha256_before", "copied_sha256",
                        "destination_sha256_after", "source_sha256_after"):
                self.assertEqual(entry[key], digest)
            self.assertEqual(entry["original_absolute"], str(self.source / entry["source_relative"]))
            self.assertEqual((self.source / entry["source_relative"]).stat().st_mtime_ns, mtime)
        self.assertEqual(freeze.verify_snapshot(self.destination), manifest)
        self.assertFalse(list(self.source.rglob("*.pyc")))

    def test_matching_snapshot_is_unchanged_on_repeat(self):
        self.create_snapshot()
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.destination.rglob("*") if p.is_file()}
        self.create_snapshot()
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.destination.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_differing_existing_file_refused_before_other_copies(self):
        copied = self.destination / "source/config.json"
        copied.parent.mkdir(parents=True)
        copied.write_bytes(b"preserve me")
        with self.assertRaisesRegex(freeze.SnapshotError, "Existing destination differs"):
            self.create_snapshot()
        self.assertEqual(copied.read_bytes(), b"preserve me")
        self.assertFalse((self.destination / "source/code.py").exists())
        self.assertFalse((self.destination / "manifest.json").exists())

    def test_matching_partial_copy_can_be_completed(self):
        copied = self.destination / "source/config.json"
        copied.parent.mkdir(parents=True)
        copied.write_bytes(self.raw)
        mtime = copied.stat().st_mtime_ns
        self.create_snapshot()
        self.assertEqual(copied.stat().st_mtime_ns, mtime)

    def test_changed_original_cannot_replace_baseline(self):
        self.create_snapshot()
        original_manifest = (self.destination / "manifest.json").read_bytes()
        (self.source / "config.json").write_bytes(b"changed input")
        with self.assertRaises(freeze.SnapshotError):
            self.create_snapshot()
        self.assertEqual((self.destination / "source/config.json").read_bytes(), self.raw)
        self.assertEqual((self.destination / "manifest.json").read_bytes(), original_manifest)

    def test_manifest_source_provenance_mismatch_refused(self):
        self.create_snapshot()
        self.git.return_value = {**self.provenance, "revision": "c" * 40}
        with self.assertRaisesRegex(freeze.SnapshotError, "provenance/selection differs"):
            self.create_snapshot()

    def test_no_manifest_if_original_changes_during_copy(self):
        copy = freeze.copy_verified

        def mutate(original, copied, expected):
            result = copy(original, copied, expected)
            if original.name == "config.json":
                original.write_bytes(b"modified during copy")
            return result

        with patch.object(freeze, "copy_verified", side_effect=mutate):
            with self.assertRaisesRegex(freeze.SnapshotError, "changed during snapshot"):
                self.create_snapshot()
        self.assertFalse((self.destination / "manifest.json").exists())

    def test_no_manifest_if_git_changes_during_copy(self):
        self.git.side_effect = [self.provenance, {**self.provenance, "status_sha256": "c" * 64}]
        with self.assertRaisesRegex(freeze.SnapshotError, "Git revision/dirty state changed"):
            self.create_snapshot()
        self.assertFalse((self.destination / "manifest.json").exists())

    def test_copy_detects_wrong_expected_hash(self):
        copied = self.destination / "config.json"
        with self.assertRaisesRegex(freeze.SnapshotError, "Copy/hash mismatch"):
            freeze.copy_verified(self.source / "config.json", copied, "0" * 64)
        self.assertFalse((self.destination / "manifest.json").exists())

    def test_offline_verifier_detects_modified_copy(self):
        self.create_snapshot()
        (self.destination / "source/config.json").write_bytes(b"damaged")
        with self.assertRaisesRegex(freeze.SnapshotError, "Snapshot file mismatch"):
            freeze.verify_snapshot(self.destination)

    def test_unmanifested_file_refused(self):
        self.create_snapshot()
        (self.destination / "extra.json").write_bytes(b"{}")
        with self.assertRaisesRegex(freeze.SnapshotError, "unmanifested"):
            freeze.verify_snapshot(self.destination)
        with self.assertRaisesRegex(freeze.SnapshotError, "outside the selected snapshot"):
            self.create_snapshot()

    def test_manifest_path_traversal_refused(self):
        manifest = self.create_snapshot()
        manifest["files"][0]["source_relative"] = "../outside.py"
        (self.destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(freeze.SnapshotError):
            freeze.verify_snapshot(self.destination)


class SelectionTests(unittest.TestCase):
    def test_selection_is_explicit_and_probe_inputs_are_optional(self):
        with tempfile.TemporaryDirectory(prefix="budget-selection-test-") as directory:
            root = Path(directory)
            required = [f"{folder}/{name}" for folder, names in freeze.RUNTIME_GROUPS.items() for name in names]
            required += [f"{freeze.HISTORICAL}/{name}" for name in freeze.HISTORICAL_SOURCES]
            required += list(freeze.FIXTURES)
            required += [f"{freeze.PROTOCOLS}/{scenario}/{name}" for scenario in freeze.SCENARIOS
                         for name in ("config.json", "forecast.json", "initial_state.json", "protocol.json", "scenario.json")]
            required += [f"{freeze.PROBES}/plant_004.json"]
            for name in required + ["outputs/old/huge.log", "deps/vendor.py", "src/rl/old.py"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"{}")
            selected, missing = freeze.select_files(root)
            self.assertEqual(set(selected), set(required))
            self.assertEqual(len(missing), 7)
            self.assertIn(f"{freeze.PROBES}/plant_069.json", missing)
            self.assertIn(f"{freeze.SAVED_DECISION}/decision_030/input.json", selected)
            (root / required[0]).unlink()
            with self.assertRaisesRegex(freeze.SnapshotError, "Missing required"):
                freeze.select_files(root)


if __name__ == "__main__":
    unittest.main()
