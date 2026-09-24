import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.visual.weights import ensure


class VisualWeightsTests(unittest.TestCase):
    def test_explicit_verified_atomic_download(self):
        data=b'verified mock weights';digest=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp,patch('training.visual.weights.SHA256',digest):
            path=Path(tmp)/'weights.pth'
            with self.assertRaises(FileNotFoundError):ensure(path)
            with patch('training.visual.weights.urllib.request.urlopen',return_value=io.BytesIO(data)):
                self.assertEqual(ensure(path,True),path.resolve())
            self.assertEqual(path.read_bytes(),data)
            self.assertEqual(ensure(path),path.resolve())
            self.assertFalse(list(Path(tmp).glob('*.partial')))
            path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'not overwritten'):ensure(path,True)
            self.assertEqual(path.read_bytes(),b'corrupt')

    def test_failed_publication_leaves_no_partial_final_weights(self):
        data=b'verified mock weights';digest=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp,patch('training.visual.weights.SHA256',digest):
            path=Path(tmp)/'weights.pth'
            with patch('training.visual.weights.urllib.request.urlopen',return_value=io.BytesIO(data)),patch(
                    'training.visual.weights.os.link',side_effect=OSError('interrupted publication')):
                with self.assertRaisesRegex(OSError,'interrupted'):ensure(path,True)
            self.assertFalse(path.exists());self.assertFalse(list(Path(tmp).glob('*.partial')))


if __name__=='__main__':unittest.main()
