import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from core import hotpot
import run_hotpotqa

ROOT = Path(__file__).resolve().parents[1]


class Model:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return {'text': next(self.outputs), 'model': 'offline-fixture',
                'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25},
                'finish_reason': 'stop'}


def config(style='standard'):
    return dict(json.loads((ROOT/'configs/hotpotqa.json').read_text()), style=style)


ITEM = {'_id': 'q', 'question': 'What is the capital of France?', 'answer': 'Paris'}
OUTPUTS = ['Lyon', 'facts\nCausal Reasoning: Consider the capital.\nBelief: B',
           'Paris', 'facts\nCausal Reasoning: Paris is the capital.\nBelief: B']


class HotpotTests(unittest.TestCase):
    def test_plain_standard_answer_does_not_require_new_terminal_marker(self):
        self.assertEqual(hotpot.parse_answer('\nParis\n'), 'Paris')

    def test_cot_preserves_original_last_line_and_prefix_parser(self):
        self.assertIn('style', inspect.signature(hotpot.parse_answer).parameters)
        self.assertEqual(hotpot.parse_answer('Step 1: reasoning.\nFinal Answer: Paris', 'cot'), 'Final  Paris')
        self.assertEqual(hotpot.parse_answer('Step 1: reasoning.\nParis\n', 'cot'), '')

    def test_same_core_selects_tie_refinement_without_gold_in_generation(self):
        model = Model(OUTPUTS)
        rec = hotpot.run_hotpot_question(ITEM, config(), model)
        self.assertEqual(rec['initial_answer'], 'Lyon')
        self.assertEqual(rec['selected_answer'], 'Paris')
        self.assertTrue(rec['refined_selected'])
        self.assertIn('Lyon', model.prompts[2])
        self.assertIn('Consider the capital.', model.prompts[2])
        changed = dict(ITEM, answer='GOLD_SENTINEL')
        other = Model(OUTPUTS)
        hotpot.run_hotpot_question(changed, config(), other)
        self.assertEqual(model.prompts, other.prompts)
        self.assertFalse(any('GOLD_SENTINEL' in p for p in other.prompts))

    def test_response_scoring_and_accounting_resume_without_new_calls(self):
        self.assertIn('judge', inspect.signature(run_hotpotqa.run).parameters)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'result.json'
            model, judge = Model(OUTPUTS), Model(['INCORRECT', 'CORRECT'])
            state = run_hotpotqa.run({'q': ITEM}, ['q'], config(), model, out,
                                    {'data': 'fixture'}, judge=judge)
            self.assertEqual(state['metrics']['baseline']['EM'], 0)
            self.assertEqual(state['metrics']['CORE']['EM'], 100)
            self.assertEqual(state['metrics']['CORE']['F1'], 100)
            self.assertEqual(state['metrics']['CORE']['Acc'], 100)
            self.assertEqual(state['metrics']['CORE']['AvgSteps'], 1)
            self.assertEqual(state['metrics']['CORE']['AvgTokens'], 100)
            self.assertEqual(state['metrics']['baseline']['AvgTokens'], 25)
            self.assertEqual(state['accounting']['judge_total_tokens'], 50)
            self.assertEqual(state['accounting']['all_total_tokens'], 150)
            self.assertIn('TRUE ANSWER: Paris', judge.prompts[0])
            self.assertIn('STUDENT ANSWER: Lyon', judge.prompts[0])
            state2 = run_hotpotqa.run({'q': ITEM}, ['q'], config(), Model([]), out,
                                     {'data': 'fixture'}, judge=Model([]))
            self.assertEqual(state, state2)

    def test_judge_failure_keeps_completed_generation_and_first_grade(self):
        self.assertIn('judge', inspect.signature(run_hotpotqa.run).parameters)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'result.json'
            with self.assertRaises(StopIteration):
                run_hotpotqa.run({'q': ITEM}, ['q'], config(), Model(OUTPUTS), out,
                                {'data': 'fixture'}, judge=Model(['INCORRECT']))
            state = run_hotpotqa.run({'q': ITEM}, ['q'], config(), Model([]), out,
                                    {'data': 'fixture'}, judge=Model(['CORRECT.']))
            # ReWOO accepts exactly CORRECT after strip; punctuation is not silently repaired.
            self.assertEqual(state['metrics']['CORE']['Acc'], 0)
            self.assertEqual(state['judge_format_failures']['CORE'], 1)
            changed = copy.deepcopy(config()); changed['judge']['api_model'] = 'different-model'
            with self.assertRaises(ValueError):
                run_hotpotqa.run({'q': ITEM}, ['q'], changed, Model([]), out,
                                {'data': 'fixture'}, judge=Model([]))

    def test_stopped_react_answer_is_reported_and_kept_in_metric_denominator(self):
        def stopped(prompt):
            return {'text': 'Agent stopped due to iteration limit or time limit.',
                    'model': 'fixture', 'finish_reason': 'max_iterations',
                    'steps': 9, 'api_calls': 8, 'tool_calls': 8,
                    'usage': {'total_tokens': 100}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'result.json'
            state = run_hotpotqa.run({'q': ITEM}, ['q'], config('react'),
                                    Model([OUTPUTS[1], OUTPUTS[3]]), out,
                                    {'data': 'fixture'}, answerer=stopped,
                                    judge=Model(['INCORRECT', 'INCORRECT']))
            self.assertEqual(state['metrics']['CORE']['EM'], 0)
            self.assertEqual(state['answer_finish_reasons']['CORE'], {'max_iterations': 1})
            run_hotpotqa.export_csv(state, {'q': ITEM}, out)
            import csv
            with out.with_suffix('.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]['answer_finish_reason'], 'max_iterations')


if __name__ == '__main__':
    unittest.main()
