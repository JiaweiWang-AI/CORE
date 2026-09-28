"""Minimal Chat Completions adapter; standard library only.

API reference: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
Credentials are read from OPENAI_API_KEY and never serialized.
"""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request
from .net import open_url

# Compatibility alias keeps transport tests independent of real network calls.
urlopen = open_url


class ChatCompletion:
    def __init__(self, config):
        self.config = config

    def __call__(self, prompt):
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('Set OPENAI_API_KEY before starting model generation.')
        body = {'model': self.config['api_model'], 'messages': [{'role': 'user', 'content': prompt}],
                **self.config['generation']}
        request = Request('https://api.openai.com/v1/chat/completions',
                          data=json.dumps(body).encode('utf-8'),
                          headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
                          method='POST')
        for attempt in range(3):
            try:
                with urlopen(request, timeout=120) as response:
                    result = json.load(response)
                break
            except HTTPError as error:
                if error.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f'Model API returned HTTP {error.code}; saved stages remain resumable.') from None
            except (URLError, TimeoutError):
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError('Model API connection failed; saved stages remain resumable.') from None
        choices = result.get('choices')
        if not isinstance(choices, list) or not choices:
            raise RuntimeError('Model API returned no completion choices; no response was saved.')
        choice = choices[0]
        if not isinstance(choice.get('message', {}).get('content'), str):
            raise RuntimeError('Model API returned no text completion; no response was saved.')
        return {'text': choice['message']['content'], 'model': result.get('model'),
                'finish_reason': choice.get('finish_reason'), 'usage': result.get('usage')}
