import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


def downloader():
    path = Path(__file__).resolve().parents[1]/'tools/fetch_hotpotqa.py'
    spec = importlib.util.spec_from_file_location('fetch_hotpotqa_retry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DownloadTests(unittest.TestCase):
    def test_transient_http_failure_retries_the_same_page(self):
        module = downloader()
        url = 'https://example.org/rows?offset=3200'
        payload = {'num_rows_total': 7405, 'rows': [{'row': {'id': 'next'}}]}
        with patch.object(module, 'open_url', side_effect=[
                HTTPError(url, 502, 'Bad Gateway', {}, None),
                io.StringIO(json.dumps(payload))]) as transport, patch('time.sleep'):
            self.assertEqual(module.read_url(url), payload)
        self.assertEqual([call.args[0] for call in transport.call_args_list], [url, url])

    def test_download_retry_limit_and_permanent_error(self):
        module = downloader()
        for status, attempts in [(502, 3), (429, 3), (404, 1)]:
            with self.subTest(status=status), patch.object(module, 'open_url',
                    side_effect=HTTPError('https://example.org', status, 'error', {}, None)) as transport, patch('time.sleep'):
                with self.assertRaises(HTTPError):
                    module.read_url('https://example.org')
                self.assertEqual(transport.call_count, attempts)

    def test_rate_limit_respects_requested_retry_delay(self):
        module = downloader()
        with patch.object(module, 'open_url', side_effect=[
                HTTPError('https://example.org', 429, 'rate limit', {'Retry-After': '9'}, None),
                io.StringIO('{"rows": []}')]), patch('time.sleep') as sleep:
            self.assertEqual(module.read_url('https://example.org'), {'rows': []})
        self.assertEqual(sleep.call_args.args, (9.0,))

    def test_public_rows_keep_ids_and_gold_but_never_import_context(self):
        path = Path(__file__).resolve().parents[1]/'tools/fetch_hotpotqa.py'
        self.assertTrue(path.exists(), 'HotpotQA input preparation is missing')
        spec = importlib.util.spec_from_file_location('fetch_hotpotqa', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        def fetch(url):
            return {'num_rows_total': 2, 'rows': [
                {'row': {'id': 'id1', 'question': 'Q1?', 'answer': 'A1', 'context': 'EXCLUDED'}, 'truncated_cells': []},
                {'row': {'id': 'id2', 'question': 'Q2?', 'answer': 'A2', 'supporting_facts': 'EXCLUDED'}, 'truncated_cells': []}]}
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'data.json'
            module.fetch_inputs(output, 'validation', 2, fetch=fetch)
            self.assertEqual(json.loads(output.read_text()), [
                {'_id': 'id1', 'question': 'Q1?', 'answer': 'A1'},
                {'_id': 'id2', 'question': 'Q2?', 'answer': 'A2'}])
            self.assertTrue(output.with_suffix('.source.json').exists())


if __name__ == '__main__':
    unittest.main()
