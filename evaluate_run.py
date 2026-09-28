"""Score a completed new run without asserting equality with historical results."""
import argparse
from pathlib import Path
from core.io import load_json, sha256
from core.scoring import compute_scores

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    run = load_json(args.run)
    if not run.get('complete'):
        raise ValueError('The run is incomplete; resume generation first.')
    data_path = args.data_root/'problems.json'
    if run['identity']['input_sha256']['problems'] != sha256(data_path):
        raise ValueError('Run annotations differ from this ScienceQA dataset.')
    problems = load_json(data_path)
    qids = run['identity']['qids']
    if (not isinstance(qids, list) or not qids or len(set(qids)) != len(qids)
            or any(q not in problems or problems[q]['split'] != 'test' for q in qids)):
        raise ValueError('Expected nonempty unique ScienceQA test IDs.')
    if set(run.get('predictions', {})) != {'GPT-3.5', 'CORE'}:
        raise ValueError('Both GPT-3.5 and CORE predictions are required.')
    for model, predictions in run['predictions'].items():
        if set(predictions) != set(qids):
            raise ValueError('Missing or unexpected predictions.')
        for q, pred in predictions.items():
            if type(pred) is not int or not 0 <= pred < len(problems[q]['choices']):
                raise ValueError('Invalid prediction for ' + q)
    # Validate every method before emitting any score.
    for model, predictions in run['predictions'].items():
        correct = sum(predictions[q] == problems[q]['answer'] for q in qids)
        print(f'{model}: {correct}/{len(qids)} = {100*correct/len(qids):.2f}% (new run)')
        if set(qids) == {q for q, p in problems.items() if p['split'] == 'test'}:
            print(compute_scores(problems, predictions))
        else:
            print('Subset only; not a Table 1 result.')


if __name__ == '__main__':
    main()
