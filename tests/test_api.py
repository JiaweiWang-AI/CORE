import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from core.api import ChatCompletion

ROOT = Path(__file__).resolve().parents[1]


class APITests(unittest.TestCase):
    def test_request_preserves_paper_generation_settings(self):
        config = json.loads((ROOT/'configs/scienceqa_A.json').read_text())
        payload = {'model': 'synthetic-snapshot', 'usage': {'total_tokens': 12},
                   'choices': [{'message': {'content': 'The answer is A.'}, 'finish_reason': 'stop'}]}
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic-test-token'}):
            with patch('core.api.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())) as send:
                result = ChatCompletion(config)('SYNTHETIC PROMPT')
        request = send.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body, {'model': 'gpt-3.5-turbo',
                               'messages': [{'role': 'user', 'content': 'SYNTHETIC PROMPT'}],
                               **config['generation']})
        self.assertEqual(result['model'], 'synthetic-snapshot')
        self.assertEqual(result['text'], 'The answer is A.')
        self.assertNotIn('synthetic-test-token', json.dumps(result))

    def test_retryable_error_recovers_without_changing_request(self):
        config = json.loads((ROOT/'configs/scienceqa_A.json').read_text())
        payload = {'choices': [{'message': {'content': 'The answer is A.'}}]}
        error = HTTPError('https://api.openai.com/v1/chat/completions', 429, 'Rate limited', {}, None)
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic-test-token'}):
            with patch('core.api.urlopen', side_effect=[error, io.BytesIO(json.dumps(payload).encode())]) as send:
                with patch('core.api.time.sleep'):
                    result = ChatCompletion(config)('SYNTHETIC PROMPT')
        self.assertEqual(result['text'], 'The answer is A.')
        self.assertEqual(send.call_args_list[0].args[0].data, send.call_args_list[1].args[0].data)

    def test_api_refusal_without_text_is_not_a_saved_answer(self):
        config = json.loads((ROOT/'configs/scienceqa_A.json').read_text())
        payload = {'choices': [{'message': {'content': None}}]}
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic-test-token'}):
            with patch('core.api.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
                with self.assertRaisesRegex(RuntimeError, 'no text completion'):
                    ChatCompletion(config)('SYNTHETIC PROMPT')

    def test_missing_key_never_sends_request(self):
        config = json.loads((ROOT/'configs/scienceqa_A.json').read_text())
        with patch.dict('os.environ', {}, clear=True), patch('core.api.urlopen') as send:
            with self.assertRaises(ValueError):
                ChatCompletion(config)('SYNTHETIC PROMPT')
            send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
