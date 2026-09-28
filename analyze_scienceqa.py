"""Compute belief-group accuracy and threshold sensitivity from a new tau=A run.

These are statistics of the supplied records, not embedded published results.
"""
import argparse
from pathlib import Path

from core.io import external_output, load_json, sha256, write_json
from core.selection import GRADES, read_belief, rebuild_predictions

ROOT = Path(__file__).resolve().parent


def analyze(problems, qids, config, records):
    fields = ('initial_outputs', 'initial_evaluations', 'refined_outputs', 'refined_evaluations')
    if config['threshold'] != 'A' or any(set(records[q]['responses']) != set(fields) for q in qids):
        raise ValueError('Threshold analysis requires complete tau=A generation for every selected question.')
    mappings = {field: {q: records[q]['responses'][field] for q in qids} for field in fields}
    counts = {q: len(problems[q]['choices']) for q in qids}
    seed = config['answer_selection_seed']
    baseline, _ = rebuild_predictions(qids, counts, mappings['initial_outputs'], seed=seed)
    groups = {g: {'count': 0, 'correct': 0} for g in GRADES}
    for q in qids:
        group = groups[read_belief(mappings['initial_evaluations'][q])]
        group['count'] += 1
        group['correct'] += int(baseline[q] == problems[q]['answer'])
    for g in groups.values():
        g['accuracy_percent'] = 100*g['correct']/g['count'] if g['count'] else None
    thresholds = {}
    for tau in GRADES:
        predictions, audit = rebuild_predictions(qids, counts, seed=seed, threshold=tau, **mappings)
        correct = sum(predictions[q] == problems[q]['answer'] for q in qids)
        thresholds[tau] = {'count': len(qids), 'correct': correct,
                           'accuracy_percent': 100*correct/len(qids),
                           'fallback_count': audit['fallback_count']}
    return {'scope': 'Statistics of supplied response records, not a historical-figure reproduction',
            'format': config['format'], 'initial_belief_groups': groups,
            'threshold_sensitivity_same_generated_candidates': thresholds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    state = load_json(args.run)
    if not state.get('complete'):
        raise ValueError('Complete the generation run before analysis.')
    path = args.data_root/'problems.json'
    if sha256(path) != state['identity']['input_sha256']['problems']:
        raise ValueError('Dataset differs from the generation input.')
    qids = state['identity']['qids']
    if not qids or len(qids) != len(set(qids)) or set(qids) != set(state['records']):
        raise ValueError('Question IDs and saved records differ.')
    report = analyze(load_json(path), qids, state['identity']['config'], state['records'])
    write_json(external_output(args.output, ROOT), report)
    print('Saved initial-belief groups and five threshold statistics for', len(qids), 'questions.')


if __name__ == '__main__':
    main()
