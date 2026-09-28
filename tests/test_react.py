import unittest
from core.react import ReactAnswerer, Wikipedia


class Model:
    def __init__(self, outputs):
        self.outputs, self.prompts = iter(outputs), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return {'text': next(self.outputs), 'model': 'fixture', 'finish_reason': 'stop',
                'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25}}


class Wiki:
    source = {'title': 'France', 'revid': 123}
    def search(self, entity):
        return 'France has capital Paris.'
    def lookup(self, term):
        return '(Result 1/1) Paris is its capital'


class ReactTests(unittest.TestCase):
    def test_finish_returns_answer_and_keeps_separate_tool_trace(self):
        model = Model(['Thought: search\nAction: Search[France]',
                       'Thought: done\nAction: Finish[Paris]'])
        result = ReactAnswerer(model, max_steps=3, wiki_factory=Wiki)('Capital of France?')
        self.assertEqual(result['text'], 'Paris')
        self.assertEqual(result['steps'], 2)
        self.assertEqual(result['tool_calls'], 1)
        self.assertEqual(result['api_calls'], 2)
        self.assertEqual(result['usage']['total_tokens'], 50)
        self.assertIn('Observation: France has capital Paris.', model.prompts[1])

    def test_iteration_limit_does_not_force_a_last_turn_answer(self):
        model = Model(['Thought: search\nAction: Search[France]',
                       'Thought: search again\nAction: Search[Paris]'])
        result = ReactAnswerer(model, max_steps=2, wiki_factory=Wiki)('A question?')
        self.assertEqual(result['text'], 'Agent stopped due to iteration limit or time limit.')
        self.assertEqual(result['steps'], 3)
        self.assertEqual(result['finish_reason'], 'max_iterations')
        self.assertFalse(any('This is the last turn' in p for p in model.prompts))

    def test_lookup_retains_upstream_sentence_iteration(self):
        def fetch(params):
            return {'query': {'pages': [{'title': 'France', 'pageid': 1,
                    'extract': 'Paris is a city. Paris is the capital.\n\nSecond paragraph.',
                    'revisions': [{'revid': 123}]}]}}
        wiki = Wikipedia(fetch=fetch)
        self.assertEqual(wiki.search('France'), 'Paris is a city. Paris is the capital.')
        self.assertEqual(wiki.lookup('Paris'), '(Result 1/2) Paris is a city')
        self.assertEqual(wiki.lookup('Paris'), '(Result 2/2)  Paris is the capital')
        self.assertEqual(wiki.lookup('Paris'), 'No More Results')


if __name__ == '__main__':
    unittest.main()
