import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.visual.weights import ensure


class VisualWeightsTests(unittest.TestCase):
    def test_read_timeout_retries_from_clean_file(self):
        class Interrupted(io.BytesIO):
            def read(self,size=-1):
                if self.tell():raise TimeoutError('read timed out')
                return super().read(4)
        data=b'verified mock weights';digest=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp,patch('training.visual.weights.SHA256',digest):
            path=Path(tmp)/'weights.pth'
            with patch('training.visual.weights.urllib.request.urlopen',side_effect=[Interrupted(data),io.BytesIO(data)]) as request:
                self.assertEqual(ensure(path,True,timeout=300,attempts=2,retry_delay=0),path.resolve())
            self.assertEqual(request.call_count,2)
            self.assertEqual(request.call_args.kwargs['timeout'],300)
            self.assertEqual(path.read_bytes(),data)
            self.assertFalse(list(Path(tmp).glob('*.partial')))

    def test_exhausted_timeouts_leave_no_weights_and_explain_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.pth'
            with patch('training.visual.weights.urllib.request.urlopen',side_effect=TimeoutError('read timed out')) as request:
                with self.assertRaisesRegex(OSError,'2 attempts.*offline'):
                    ensure(path,True,attempts=2,retry_delay=0)
            self.assertEqual(request.call_count,2)
            self.assertFalse(path.exists());self.assertFalse(list(Path(tmp).glob('*.partial')))

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
