import hashlib
from pathlib import Path
import tempfile
import unittest

from inference.experts.assets import model_identity, verify_asset


class AssetTests(unittest.TestCase):
    def test_hash_and_provenance_are_required(self):
        with tempfile.TemporaryDirectory(dir='runs') as directory:
            path = Path(directory) / 'fixture.bin'
            path.write_bytes(b'synthetic artifact, not weights')
            spec = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        source='synthetic unittest', license='test-only')
            verified = verify_asset(spec)
            self.assertEqual(Path(verified['path']), path.resolve())
            for update in ({'sha256': '0' * 64}, {'source': ''}, {'license': ''}, {'extra': True}):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    verify_asset(dict(spec, **update))
            path.unlink()
            with self.assertRaises((ValueError, OSError)):
                verify_asset(spec)

    def test_identity_is_portable_but_binds_settings_and_source(self):
        asset = dict(path='one', sha256='a' * 64, source='source', license='license')
        identity = model_identity('test', asset, {'threshold': 0.5})
        self.assertEqual(identity, model_identity('test', dict(asset, path='two'), {'threshold': 0.5}))
        self.assertNotEqual(identity, model_identity('test', asset, {'threshold': 0.6}))
        self.assertNotEqual(identity, model_identity('test', dict(asset, sha256='b' * 64), {'threshold': 0.5}))
