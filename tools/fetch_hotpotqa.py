"""Fetch public HotpotQA question-only inputs with IDs and a source receipt.

Uses the HotpotQA organization's Hugging Face dataset. This prepares a new
evaluation sample; it does not identify the authors' historical Table 2 sample.
"""
import argparse
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.io import external_output, sha256, write_json
from core.net import open_url

DATASET = 'hotpotqa/hotpot_qa'


def read_url(url):
    for attempt in range(3):
        delay = 2 ** attempt
        try:
            with open_url(url, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            if error.code == 429:
                delay = 5 * 2 ** attempt
            retry_after = error.headers.get('Retry-After')
            if retry_after:
                try:
                    requested_delay = float(retry_after)
                except ValueError:
                    requested_delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
                delay = max(delay, requested_delay)
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(delay)


def fetch_inputs(output, split='validation', limit=2, fetch=read_url):
    if split not in ('train', 'validation') or limit < 0:
        raise ValueError('Choose train/validation and a nonnegative limit (0 means all).')
    output = external_output(output, Path(__file__).resolve().parents[1])
    items, urls, total = [], [], None
    while total is None or len(items) < (min(limit, total) if limit else total):
        if items:
            time.sleep(1)  # Pace pagination to avoid bursts against the public service.
        remaining = (limit - len(items)) if limit else 100
        url = 'https://datasets-server.huggingface.co/rows?' + urlencode({
            'dataset': DATASET, 'config': 'fullwiki', 'split': split,
            'offset': len(items), 'length': min(100, remaining)})
        payload = fetch(url)
        total = payload['num_rows_total']
        rows = payload['rows']
        if not rows:
            raise ValueError('Dataset service returned an incomplete range.')
        urls.append(url)
        for row in rows:
            data = row['row']
            if {'id', 'question', 'answer'}.intersection(row.get('truncated_cells', [])):
                raise ValueError('A required question/answer field was truncated.')
            if any(not isinstance(data.get(k), str) for k in ('id', 'question', 'answer')):
                raise ValueError('Expected string IDs, questions and reference answers.')
            items.append({'_id': data['id'], 'question': data['question'], 'answer': data['answer']})
        print(f'Fetched {len(items)}/{min(limit, total) if limit else total} questions.', flush=True)
    if len({x['_id'] for x in items}) != len(items):
        raise ValueError('Dataset service returned duplicate IDs.')
    write_json(output, items)
    write_json(output.with_suffix('.source.json'), {
        'dataset': DATASET, 'configuration': 'fullwiki', 'split': split,
        'urls': urls, 'selection': 'First N rows in the public dataset; sampling is performed by run_hotpotqa.py.',
        'count': len(items), 'available_rows': total, 'sha256': sha256(output),
        'transform': 'id renamed to _id; retain question/answer; omit context/supporting facts.',
        'license': 'CC BY-SA 4.0; https://hotpotqa.github.io/'})
    return items


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--limit', type=int, default=2, help='Download first N rows; use 0 for the complete split.')
    args = parser.parse_args()
    fetch_inputs(args.output, args.split, args.limit)
