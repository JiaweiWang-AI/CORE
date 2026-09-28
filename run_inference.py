"""Generate new ScienceQA responses using supplied captions and a model API.

This is a new run, not a bitwise recreation of a historical API response.
"""
import argparse
from pathlib import Path
import sys

from core.api import ChatCompletion
from core.io import external_output, load_json, sha256
from core.pipeline import run_experiment

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['A', 'AE', 'ALE'], default='ALE')
    parser.add_argument('--data-root', type=Path, required=True,
                        help='External directory containing problems.json and pid_splits.json.')
    parser.add_argument('--captions', type=Path, required=True,
                        help='JSON containing {"captions": {question_id: caption_text}}.')
    parser.add_argument('--output', type=Path, required=True, help='Checkpoint outside CORE/.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--limit', type=int, default=1, help='First N test questions; default 1.')
    group.add_argument('--full-test', action='store_true', help='All 4,241 questions; up to 16,964 API calls.')
    parser.add_argument('--threshold', choices=['A', 'B', 'C', 'D', 'F'])
    parser.add_argument('--model', help='Explicit API model override; recorded as a new-run setting.')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print plan without API calls.')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive; use --full-test for all questions.')
    output = external_output(args.output, ROOT)
    config = load_json(ROOT/f'configs/scienceqa_{args.format}.json')
    if args.threshold:
        config['threshold'] = args.threshold
    if args.model:
        config['api_model'] = args.model
    problems_path = args.data_root/'problems.json'
    splits_path = args.data_root/'pid_splits.json'
    problems, splits = load_json(problems_path), load_json(splits_path)
    qids = splits['test'] if args.full_test else splits['test'][:args.limit]
    if not qids or len(set(qids)) != len(qids) or any(problems[q]['split'] != 'test' for q in qids):
        raise ValueError('Expected nonempty, unique ScienceQA test IDs.')
    if args.full_test and len(qids) != 4241:
        raise ValueError('The official full ScienceQA test split has 4,241 questions.')
    captions = load_json(args.captions)['captions']
    if not isinstance(captions, dict) or any(not isinstance(v, str) for v in captions.values()):
        raise ValueError('Caption values must be strings.')
    needed = qids + config['shot_qids']
    missing = [q for q in needed if problems[q]['image'] and not captions.get(q, '').strip()]
    if missing:
        raise ValueError(f'Missing captions for {len(missing)} image questions; first: {missing[:10]}')
    for q, problem in problems.items():
        problem['caption'] = captions.get(q, '')
    identity = {'problems': sha256(problems_path), 'splits': sha256(splits_path),
                'captions': sha256(args.captions)}
    identity.update({f'code/{p.name}': sha256(p) for p in sorted((ROOT/'core').glob('*.py'))})
    identity['code/run_inference.py'] = sha256(Path(__file__))
    print(f"New run: {config['api_model']}; QCM-{args.format}; threshold {config['threshold']}; "
          f'{len(qids)} questions; up to {4*len(qids)} model calls.')
    print('Caption SHA256:', identity['captions'])
    if args.dry_run:
        print('DRY RUN: input validation passed; no model call or output write.')
        return 0
    state = run_experiment(problems, qids, config, ChatCompletion(config), output, identity)
    print(f"Saved {len(state['records'])} questions to {output}")
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        raise SystemExit(1)
