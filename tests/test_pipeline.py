import copy
import json
from pathlib import Path
import tempfile
import unittest

from core.pipeline import run_question, run_experiment

ROOT = Path(__file__).resolve().parents[1]


def fixtures(threshold='A'):
    config = json.loads((ROOT/'configs/scienceqa_A.json').read_text())
    config['threshold'] = threshold
    problems = {}
    for q in ['17099', '6960', 'q']:
        problems[q] = dict(question='Synthetic question?', hint='', caption='',
                           choices=['First', 'Second'], answer=0, lecture='TRAIN LECTURE',
                           solution='TRAIN SOLUTION', split='test' if q == 'q' else 'train')
    return problems, config


class FakeModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return {'text': next(self.outputs), 'model': 'synthetic-model',
                'usage': {'total_tokens': 1}, 'finish_reason': 'stop'}


class PipelineTests(unittest.TestCase):
    def test_invalid_evaluation_never_poisons_checkpoint(self):
        p, c = fixtures()
        sequences = [(['The answer is A.', 'facts\nBelief: E'], {'initial_outputs'}),
                     (['The answer is A.', 'facts\nBelief: D', 'The answer is B.', 'facts\nBelief: E'],
                      {'initial_outputs', 'initial_evaluations', 'refined_outputs'})]
        for outputs, saved_stages in sequences:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'run.json'
                with self.assertRaises(ValueError):
                    run_experiment(p, ['q'], c, FakeModel(outputs), path, {'data': 'synthetic'})
                saved = json.loads(path.read_text())['records']['q']['responses']
                self.assertEqual(set(saved), saved_stages)
                self.assertFalse(any('Belief: E' in text for text in saved.values()))

    def test_four_stage_order_and_original_normalization(self):
        p, c = fixtures()
        model = FakeModel(['  The answer is A.  ', 'facts\n\nBelief: A',
                           'The answer is B.', 'facts\nBelief: A'])
        record = run_question(p, 'q', c, model)
        self.assertEqual(len(model.prompts), 4)
        self.assertEqual(record['responses']['initial_evaluations'],
                         'Observed Information: facts\nBelief: A')
        self.assertIn('The answer is A.', model.prompts[1])
        self.assertIn('The answer is B.', model.prompts[3])
        # The refinement context excludes the terminal Belief line.
        self.assertNotIn('Belief: A', model.prompts[2])

    def test_C_threshold_skips_high_belief(self):
        p, c = fixtures('C')
        model = FakeModel(['The answer is A.', 'facts\nBelief: A'])
        record = run_question(p, 'q', c, model)
        self.assertEqual(len(model.prompts), 2)
        self.assertNotIn('refined_outputs', record['responses'])

    def test_test_targets_cannot_change_prompts(self):
        p, c = fixtures()
        outputs = ['The answer is A.', 'facts\nBelief: A',
                   'The answer is B.', 'facts\nBelief: B']
        a = FakeModel(outputs)
        run_question(p, 'q', c, a)
        changed = copy.deepcopy(p)
        changed['q'].update(answer=1, lecture='SECRET TEST LECTURE',
                            solution='SECRET TEST SOLUTION')
        b = FakeModel(outputs)
        run_question(changed, 'q', c, b)
        self.assertEqual(a.prompts, b.prompts)
        self.assertFalse(any('SECRET TEST' in text for text in b.prompts))

    def test_resume_keeps_completed_calls_and_rebuilds_predictions(self):
        p, c = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run.json'
            model = FakeModel(['unparsed', 'facts\nBelief: A'])
            with self.assertRaises(StopIteration):
                run_experiment(p, ['q'], c, model, path, {'data': 'synthetic'})
            saved = json.loads(path.read_text())
            self.assertEqual(set(saved['records']['q']['responses']),
                             {'initial_outputs', 'initial_evaluations'})
            model = FakeModel(['The answer is A.', 'facts\nBelief: D'])
            result = run_experiment(p, ['q'], c, model, path, {'data': 'synthetic'})
            self.assertEqual(len(model.prompts), 2)
            self.assertEqual(result['predictions']['CORE'], {'q': 1})
            self.assertEqual(result['audits']['CORE']['fallback_count'], 1)
            model = FakeModel([])
            result2 = run_experiment(p, ['q'], c, model, path, {'data': 'synthetic'})
            self.assertEqual(result, result2)
            self.assertEqual(model.prompts, [])

    def test_resume_rejects_different_configuration_or_inputs(self):
        p, c = fixtures('C')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run.json'
            run_experiment(p, ['q'], c, FakeModel(['The answer is A.', 'facts\nBelief: A']),
                           path, {'data': 'one'})
            changed = copy.deepcopy(c)
            changed['threshold'] = 'A'
            for config, identity in [(changed, {'data': 'one'}), (c, {'data': 'two'})]:
                with self.assertRaises(ValueError):
                    run_experiment(p, ['q'], config, FakeModel([]), path, identity)


if __name__ == '__main__':
    unittest.main()
