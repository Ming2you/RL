from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from work.research_snapshot import build, inspect_or_restore, safe_target


class ResearchSnapshotTests(unittest.TestCase):
    def test_round_trip_preserves_bytes_and_never_overwrites_changes(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp) / 'source'
            (root / 'results/run').mkdir(parents=True)
            (root / 'data').mkdir()
            payload = b'checkpoint\x00\xff\r\n'
            (root / 'results/run/checkpoint.pkl').write_bytes(payload)
            (root / 'results/run/STOP').write_text('paused')
            (root / 'data/transitions.npz').write_bytes(b'data\x01')
            bundle = Path(tmp) / 'bundle'
            build(root, bundle)
            inspect_or_restore(root, bundle, 'verify-local')
            fresh = Path(tmp) / 'clone'
            fresh.mkdir()
            inspect_or_restore(fresh, bundle, 'restore')
            self.assertEqual((fresh / 'results/run/checkpoint.pkl').read_bytes(), payload)
            self.assertTrue((fresh / 'results/run/STOP').exists())
            inspect_or_restore(fresh, bundle, 'restore')
            (fresh / 'data/transitions.npz').write_bytes(b'user-change')
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                inspect_or_restore(fresh, bundle, 'restore')
            self.assertEqual((fresh / 'data/transitions.npz').read_bytes(), b'user-change')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                build(root, bundle)

    def test_rejects_escape_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('../outside', 'results/../../outside', 'C:/outside', '/results/x',
                         'results/x:stream', 'results\\x', '.git/config'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_target(Path(tmp), name)

    def test_tampered_archive_part_fails_before_restore(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            repo = Path(tmp) / 'repo'
            (repo / 'data').mkdir(parents=True)
            (repo / 'data/item').write_bytes(b'original')
            bundle = Path(tmp) / 'bundle'
            build(repo, bundle)
            part = bundle / 'research.zip.part001'
            part.write_bytes(part.read_bytes() + b'tampered')
            target = Path(tmp) / 'fresh'
            target.mkdir()
            with self.assertRaisesRegex(ValueError, 'split part hash mismatch'):
                inspect_or_restore(target, bundle, 'restore')
            self.assertEqual(list(target.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
