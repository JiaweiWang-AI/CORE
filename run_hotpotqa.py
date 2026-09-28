"""Run the shared CORE algorithm on an external HotpotQA file.

Generate responses and evaluate baseline/CORE with EM, F1 and model-judged Acc.
Explicit new-run configuration; no historical Table 2 score claim.
"""
import argparse
from collections import Counter
import copy
import csv
from pathlib import Path
import random
import sys

from core.api import ChatCompletion
from core.hotpot import run_hotpot_question
from core.hotpot_evaluation import grade_answer, summarize, tokens
from core.react import ReactAnswerer
from core.io import external_output, load_json, sha256, write_json

ROOT = Path(__file__).resolve().parent


def run(items, qids, config, complete, checkpoint, input_hashes, answerer=None, judge=None):
    if not qids or len(set(qids)) != len(qids):
        raise ValueError('Expected a nonempty unique sample.')
    judge = judge or ChatCompletion(config['judge'])
    identity = {'config': config, 'qids': qids, 'input_sha256': input_hashes}
    if checkpoint.exists():
        state = load_json(checkpoint)
        if state.get('identity') != identity:
            raise ValueError('Checkpoint inputs, sample order, code or configuration differ.')
        if set(state['records']) - set(qids):
            raise ValueError('Checkpoint contains unexpected question IDs.')
    else:
        state = {'identity': identity, 'records': {}, 'complete': False}
    for qid in qids:
        def save(record):
            state['records'][qid] = record
            write_json(checkpoint, state)
        record = run_hotpot_question(items[qid], config, complete,
                                     state['records'].get(qid), save, answerer)
        save(record)
        grades = record.setdefault('evaluations', {})
        for method, field in [('baseline', 'initial_answer'), ('CORE', 'selected_answer')]:
            if method not in grades:
                grades[method] = grade_answer(items[qid]['question'], record[field],
                                              items[qid]['answer'], judge)
                save(record)
    state['predictions'] = {}
    state['parse_failures'] = {}
    state['answer_finish_reasons'] = {}
    for method, field in [('baseline', 'initial_answer'), ('CORE', 'selected_answer')]:
        predictions = {q: state['records'][q][field] or '' for q in qids}
        state['predictions'][method] = predictions
        state['parse_failures'][method] = sum(not state['records'][q][field].strip() for q in qids)
        state['answer_finish_reasons'][method] = dict(Counter(
            state['records'][q]['calls'][
                'initial_outputs' if method == 'baseline' else state['records'][q]['selected_stage']
            ].get('finish_reason') or 'unknown' for q in qids))
    state['metrics'], state['judge_format_failures'], state['accounting'] = summarize(state['records'], qids)
    state['metric_definitions'] = {
        'EM_F1_Acc': 'Percent, 0 to 100; Acc uses the saved QAEval prompt and strict CORRECT output.',
        'AvgSteps': 'Steps of selected answer trajectory: Standard=1, CoT=count(Step), ReAct=intermediate actions+1.',
        'AvgAnswerStepsTotal': 'Sum of initial and executed refinement answer trajectory steps.',
        'AvgTokens': 'Baseline=initial answer; CORE=all executed CORE stages including initial answer. Judge excluded.',
        'AvgAPICalls': 'Generation API completions only. ReAct includes each agent iteration; judge reported separately.',
        'historical_scope': 'Explicit accounting for this run; historical FR aggregation was not recovered.'}
    state['complete'] = True
    write_json(checkpoint, state)
    return state


def export_csv(state, items, output):
    """Readable per-question responses and metrics, alongside the complete JSON trace."""
    fields = ['id', 'method', 'question', 'answer', 'response', 'preds', 'em', 'f1', 'acc',
              'steps', 'total_tokens', 'answer_finish_reason', 'judge_grade', 'judge_model', 'judge_tokens']
    with output.with_suffix('.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in state['identity']['qids']:
            rec = state['records'][qid]
            for method in ('baseline', 'CORE'):
                stage = 'initial_outputs' if method == 'baseline' else rec['selected_stage']
                g = rec['evaluations'][method]
                calls = [rec['calls']['initial_outputs']] if method == 'baseline' else rec['calls'].values()
                writer.writerow(dict(id=qid, method=method, question=items[qid]['question'],
                    answer=items[qid]['answer'], response=rec['responses'][stage], preds=g['prediction'],
                    em=g['em'], f1=g['f1'], acc=g['acc'], steps=rec['calls'][stage]['steps'],
                    total_tokens=sum(tokens(c) for c in calls),
                    answer_finish_reason=rec['calls'][stage].get('finish_reason'), judge_grade=g['grade'],
                    judge_model=g['model'], judge_tokens=tokens(g)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Official HotpotQA dev JSON list.')
    parser.add_argument('--style', choices=['standard', 'cot', 'react', 'all'],
                        help='Override config style; packaged default standard.')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/hotpotqa.json')
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sample-size', type=int, default=1, help='New random sample size; default 1.')
    group.add_argument('--ids', type=Path, help='JSON list of explicit question IDs, preserving order.')
    parser.add_argument('--sample-seed', type=int, default=1)
    parser.add_argument('--threshold', choices=['A', 'B', 'C', 'D', 'F'],
                        help='Override config threshold; packaged default A.')
    parser.add_argument('--model')
    parser.add_argument('--judge-model')
    parser.add_argument('--temperature', type=float, help='Override answer generation temperature.')
    parser.add_argument('--max-tokens', type=int, help='Override answer generation limit.')
    parser.add_argument('--react-max-steps', type=int, help='Maximum ReAct iterations (1..30).')
    parser.add_argument('--react-examples', type=int, choices=range(7), help='Use first N pinned upstream examples; packaged default 1.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = external_output(args.output, ROOT)
    raw = load_json(args.data)
    if not isinstance(raw, list) or not raw:
        raise ValueError('Expected a nonempty official HotpotQA JSON list.')
    items = {item['_id']: item for item in raw}
    if len(items) != len(raw):
        raise ValueError('Duplicate question IDs in dataset.')
    if args.ids:
        qids = load_json(args.ids)
    else:
        if not 1 <= args.sample_size <= len(items):
            raise ValueError('Sample size must be positive and within the dataset size.')
        qids = random.Random(args.sample_seed).sample(list(items), args.sample_size)
    if (not isinstance(qids, list) or not qids or len(qids) != len(set(qids)) or
            any(q not in items or not isinstance(items[q].get('answer'), str) for q in qids)):
        raise ValueError('Expected unique dataset IDs with answer annotations for evaluation.')
    if any(not isinstance(items[q].get('question'), str) or not items[q]['question'].strip() for q in qids):
        raise ValueError('Every selected item needs a nonempty question.')
    config = load_json(args.config)
    config['sample_seed'] = args.sample_seed
    if args.threshold is not None:
        config['threshold'] = args.threshold
    requested_style = args.style if args.style is not None else config['style']
    if config.get('context_mode') != 'none':
        raise ValueError('This paper-aligned entry point uses question-only input.')
    if args.model:
        config['api_model'] = args.model
    if args.judge_model:
        config['judge']['api_model'] = args.judge_model
    for field in ('temperature', 'max_tokens'):
        value = getattr(args, field)
        if value is not None:
            config['generation'][field] = config['react_generation'][field] = value
    for field in ('react_max_steps', 'react_examples'):
        if getattr(args, field) is not None:
            config[field] = getattr(args, field)
    if not 1 <= config['react_max_steps'] <= 30 or not 0 <= config['react_examples'] <= 6:
        raise ValueError('Invalid ReAct iteration or example count.')
    hashes = {'data': sha256(args.data), 'runner': sha256(Path(__file__))}
    hashes.update({f'core/{p.name}': sha256(p) for p in sorted((ROOT/'core').glob('*.py'))})
    styles = ['standard', 'cot', 'react'] if requested_style == 'all' else [requested_style]
    print(f"HotpotQA: {len(qids)} questions; {styles}; question-only; tau={config['threshold']}.", flush=True)
    if args.dry_run:
        print('DRY RUN: no API calls, no output write. Sample IDs will be recorded in the run.')
        return
    combined = {'qids': qids, 'runs': {}}
    for style in styles:
        current = copy.deepcopy(config)
        current['style'] = style
        complete = ChatCompletion(dict(current, generation=current['evaluation_generation']))
        if style == 'react':
            answerer = ReactAnswerer(ChatCompletion(dict(current, generation=current['react_generation'])),
                                     max_steps=current['react_max_steps'], examples=current['react_examples'])
        else:
            answerer = ChatCompletion(current)
        target = output.with_name(output.stem + '.' + style + '.json') if requested_style == 'all' else output
        state = run(items, qids, current, complete, target, hashes, answerer, ChatCompletion(current['judge']))
        export_csv(state, items, target)
        combined['runs'][style] = {'file': str(target), 'metrics': state['metrics'], 'accounting': state['accounting'],
                                   'answer_finish_reasons': state['answer_finish_reasons'],
                                   'parse_failures': state['parse_failures'],
                                   'judge_format_failures': state['judge_format_failures']}
        for method, metrics in state['metrics'].items():
            print(style, method, metrics, flush=True)
            print('  answer finish reasons:', state['answer_finish_reasons'][method], flush=True)
    if requested_style == 'all':
        write_json(output, combined)
    print('Saved response traces, per-question CSV and aggregate metrics for this new run.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        raise SystemExit(1)
