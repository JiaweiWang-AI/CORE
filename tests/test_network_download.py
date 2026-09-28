import hashlib
import io
from pathlib import Path
import ssl
import tempfile
import unittest
from unittest.mock import patch

from core.net import tls_context, download_file


class NetworkTests(unittest.TestCase):
    def test_tls_verifies_certificates_and_loads_trust(self):
        context = tls_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertGreater(context.cert_store_stats()['x509_ca'], 0)

    def test_download_is_verified_and_rerunnable(self):
        content = b'synthetic download'
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'file.json'
            with patch('core.net.open_url', return_value=io.BytesIO(content)) as request:
                download_file('https://example.org/file', path, digest)
                download_file('https://example.org/file', path, digest)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(path.read_bytes(), content)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                download_file('https://example.org/file', path, digest)

    def test_bad_hash_never_becomes_a_complete_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'file.json'
            with patch('core.net.open_url', return_value=io.BytesIO(b'bad')):
                with self.assertRaises(ValueError):
                    download_file('https://example.org/file', path, '0'*64)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
