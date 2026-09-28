import random
import unittest

from core.selection import (GRADES, needs_refinement, prefer_refinement,
                             rebuild_predictions)


def evaluation(grade):
    return f'Observed Information: synthetic\nCausal Reasoning: synthetic\nBelief: {grade}'


class ReplayTests(unittest.TestCase):
    def test_paper_threshold_and_early_threshold(self):
        self.assertEqual([needs_refinement(g, 'A') for g in GRADES], [True]*5)
        self.assertEqual([needs_refinement(g, 'C') for g in GRADES],
                         [False, False, True, True, True])

    def test_algorithm_one_ties_better_and_worse(self):
        self.assertTrue(prefer_refinement('A', 'A'))
        self.assertTrue(prefer_refinement('D', 'B'))
        self.assertFalse(prefer_refinement('A', 'D'))
        self.assertTrue(prefer_refinement('F', 'F'))

    def test_selected_text_is_parsed_before_fallback(self):
        qids=['one', 'two', 'three']
        result, audit = rebuild_predictions(qids, dict.fromkeys(qids, 2),
            {'one':'The answer is A.', 'two':'The answer is (A).', 'three':'The answer is A.'},
            initial_evaluations={'one':evaluation('A'), 'two':evaluation('A'), 'three':evaluation('D')},
            refined_outputs={'one':'The answer is B.', 'two':'The answer is A.', 'three':'The answer is B.'},
            refined_evaluations={'one':evaluation('A'), 'two':evaluation('D'), 'three':evaluation('A')})
        # q1: equal scores -> refined B; q2: lower score -> initial malformed
        # text -> first random binary choice with seed 5 is B; q3: improved B.
        self.assertEqual(result, {'one':1, 'two':1, 'three':1})
        self.assertEqual(audit['fallback_ids'], ['two'])

    def test_fallback_rng_is_independent_of_global_random_state(self):
        qids=['first','valid','second','third']
        texts={'first':'N/A','valid':'The answer is A.',
               'second':'The answer is (B).','third':'The answer is N/A.'}
        state=random.getstate()
        result, audit=rebuild_predictions(qids,dict.fromkeys(qids,2),texts)
        self.assertEqual(result,{'first':1,'valid':0,'second':1,'third':0})
        self.assertEqual(audit['fallback_count'],3)
        self.assertEqual(random.getstate(),state)
        self.assertEqual(rebuild_predictions(qids,dict.fromkeys(qids,2),texts)[0],result)

    def test_missing_refinement_is_allowed_only_below_execution_gate(self):
        kwargs=dict(initial_evaluations={'q':evaluation('A')},refined_outputs={},refined_evaluations={})
        self.assertEqual(rebuild_predictions(['q'],{'q':2},{'q':'The answer is A.'},threshold='C',**kwargs)[0],{'q':0})
        with self.assertRaises(KeyError):
            rebuild_predictions(['q'],{'q':2},{'q':'The answer is A.'},threshold='A',**kwargs)

    def test_duplicate_ids_invalid_counts_and_invalid_grades_are_rejected(self):
        with self.assertRaises(ValueError):
            rebuild_predictions(['q','q'],{'q':2},{'q':'The answer is A.'})
        with self.assertRaises(ValueError):
            rebuild_predictions(['q'],{'q':0},{'q':'The answer is A.'})
        with self.assertRaises(ValueError):
            needs_refinement('E')

    def test_partial_core_inputs_are_not_silently_ignored(self):
        for name in ('initial_evaluations', 'refined_outputs', 'refined_evaluations'):
            with self.assertRaises(ValueError):
                rebuild_predictions(['q'], {'q': 2}, {'q': 'The answer is A.'}, **{name: {}})


if __name__ == '__main__':
    unittest.main()
