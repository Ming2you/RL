import json
from pathlib import Path
import tempfile
import unittest

from work import research_snapshot as archive
from work.sdmpc_rl_handoff_20260930 import FAMILIES, select


class HandoffTest(unittest.TestCase):
    def test_selected_roundtrip_preserves_conflicts_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'original'
            slot = root / 'results' / FAMILIES[0]
            (slot / 'checkpoints').mkdir(parents=True)
            (slot / 'STOP').write_text('retain')
            (slot / 'checkpoints' / 'old.pt').write_bytes(b'old')
            (slot / 'checkpoints' / 'latest.pt').write_bytes(b'latest')
            (slot / 'latest.json').write_text(json.dumps({'path': 'checkpoints/latest.pt'}))
            selected, omitted = select(root.resolve())
            self.assertEqual(len(selected), 3)
            self.assertEqual(len(omitted), 1)
            bundle = Path(temp) / 'bundle'
            archive.build(root, bundle, selected=selected, status='HANDOFF')
            restored = Path(temp) / 'restored'
            archive.inspect_or_restore(restored, bundle, 'restore')
            archive.inspect_or_restore(restored, bundle, 'verify-local')
            stop = restored / 'results' / FAMILIES[0] / 'STOP'
            self.assertEqual(stop.read_text(), 'retain')
            stop.write_text('different')
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                archive.inspect_or_restore(restored, bundle, 'restore')

    def test_selected_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                archive.build(root, root / 'bundle', selected=['results/../../secret'])
            self.assertFalse((root / 'bundle').exists())

    def test_missing_referenced_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slot = root / 'results' / FAMILIES[0]
            slot.mkdir(parents=True)
            (slot / 'latest.json').write_text(json.dumps({'path': 'checkpoints/missing.pt'}))
            with self.assertRaisesRegex(ValueError, 'missing'):
                select(root.resolve())


if __name__ == '__main__':
    unittest.main()
