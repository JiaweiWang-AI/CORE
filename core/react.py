"""ReWOO/LangChain ReAct protocol with a current Wikipedia transport.

Prompt text, last-line action parsing, sentence lookup, and step counting follow
the pinned upstreams in PROVENANCE.json. MediaWiki JSON replaces the old
wikipedia package transport; it is not an archived Wikipedia snapshot.
"""
import json
import re
from urllib.parse import urlencode
from urllib.request import Request
from .net import open_url
from .hotpot_prompts import REACT_INSTRUCTION, REACT_EXAMPLES


def fetch_wikipedia(params):
    query = {'action': 'query', 'format': 'json', 'formatversion': 2, **params}
    request = Request('https://en.wikipedia.org/w/api.php?' + urlencode(query),
                      headers={'User-Agent': 'CORE-research-reproduction/1.0 (https://doi.org/10.1145/3664647.3681263)'})
    with open_url(request, timeout=60) as response:
        data = json.load(response)
    if 'error' in data:
        raise RuntimeError('Wikipedia API error: ' + str(data['error'].get('code')))
    return data


class Wikipedia:
    def __init__(self, fetch=fetch_wikipedia):
        self.fetch = fetch
        self.page, self.source = None, {}
        self.lookup_term, self.lookup_index = '', 0

    def search(self, entity):
        self.page, self.source = None, {}
        data = self.fetch({'titles': entity, 'redirects': 1, 'prop': 'extracts|revisions|pageprops',
                           'explaintext': 1, 'rvprop': 'ids', 'ppprop': 'disambiguation'})
        pages = data.get('query', {}).get('pages', [])
        page = pages[0] if pages else {}
        if page.get('missing') or not page.get('extract') or 'disambiguation' in page.get('pageprops', {}):
            suggestions = self.fetch({'list': 'search', 'srsearch': entity, 'srlimit': 5})
            titles = [x['title'] for x in suggestions.get('query', {}).get('search', [])]
            return 'Could not find [' + entity + ']. Similar: ' + repr(titles)
        self.page = page['extract']
        self.source = {'title': page['title'], 'pageid': page['pageid'],
                       'revid': (page.get('revisions') or [{}])[0].get('revid'),
                       'transport': 'MediaWiki JSON extracts'}
        return self.page.split('\n\n')[0]

    def lookup(self, keyword):
        if self.page is None:
            raise ValueError('Cannot lookup without a successful search first')
        if keyword.lower() != self.lookup_term:
            self.lookup_term, self.lookup_index = keyword.lower(), 0
        else:
            self.lookup_index += 1
        matches = [s for s in self.page.split('.') if self.lookup_term in s.lower()]
        if not matches:
            return 'No Results'
        if self.lookup_index >= len(matches):
            return 'No More Results'
        return f'(Result {self.lookup_index + 1}/{len(matches)}) {matches[self.lookup_index]}'


class ReactAnswerer:
    def __init__(self, complete, max_steps=8, wiki_factory=Wikipedia, examples=0):
        if not 1 <= max_steps <= 30 or not 0 <= examples <= len(REACT_EXAMPLES):
            raise ValueError('ReAct requires 1..30 iterations and 0..6 examples.')
        self.complete, self.max_steps, self.wiki_factory = complete, max_steps, wiki_factory
        self.examples = examples

    def __call__(self, task):
        wiki = self.wiki_factory()
        prefix = REACT_INSTRUCTION
        if self.examples:
            prefix += 'Here are some examples.\n\n' + '\n\n'.join(REACT_EXAMPLES[:self.examples])
        scratchpad, trace, usage = '', [], {}
        tools, actions, answer = 0, 0, None
        for step in range(1, self.max_steps + 1):
            prompt = prefix + '\n\nQuestion: ' + task + '\n' + scratchpad
            result = self.complete(prompt)
            text = result['text']
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if key in (result.get('usage') or {}):
                    usage[key] = usage.get(key, 0) + result['usage'][key]
            # LangChain 0.0.187 ReActOutputParser reads the last line only.
            line = text.strip().split('\n')[-1]
            if not line.startswith('Action: '):
                raise ValueError('ReAct response has no terminal Action: line; resume to retry this stage.')
            match = re.search(r'(.*?)\[(.*?)\]', line[len('Action: '):])
            if match is None:
                raise ValueError('ReAct response has an invalid action directive; resume to retry this stage.')
            action, argument = match.groups()
            entry = {'step': step, 'prompt': prompt, 'model_text': text,
                     'model': result.get('model'), 'usage': result.get('usage'),
                     'finish_reason': result.get('finish_reason'), 'action': action, 'argument': argument}
            trace.append(entry)
            if action == 'Finish':
                answer = argument
                break
            actions += 1
            if action == 'Search':
                observation = wiki.search(argument)
                tools += 1
            elif action == 'Lookup':
                observation = wiki.lookup(argument)
                tools += 1
            else:
                observation = f'{action} is not a valid tool, try one of [Search, Lookup].'
            entry.update(observation=observation, source=dict(getattr(wiki, 'source', {})))
            scratchpad += text + '\nObservation: ' + observation + '\nThought:'
        return {'text': answer if answer is not None else 'Agent stopped due to iteration limit or time limit.',
                'model': trace[-1]['model'], 'usage': usage, 'steps': actions + 1,
                'finish_reason': 'finish' if answer is not None else 'max_iterations',
                'tool_trace': trace, 'tool_calls': tools, 'api_calls': len(trace)}
