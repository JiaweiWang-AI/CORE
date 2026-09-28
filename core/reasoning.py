"""The shared CORE loop for multiple-choice and open-answer tasks."""
import hashlib
from .selection import GRADES, needs_refinement, read_belief

STAGES = ('initial_outputs', 'initial_evaluations', 'refined_outputs', 'refined_evaluations')


def run_reasoning(qa_prompt, evaluation_prompt, refinement_prompt, threshold, complete,
                  record=None, save=None, answerer=None, preserve_answer_whitespace=False,
                  record_prompts=False):
    """Prompt adapters supply task text; all tasks execute the same refinement gate.

    answerer is optionally a tool-enabled QA backend. It is used for both the
    initial answer and refinement; complete is used for the two evaluations.
    """
    if threshold not in GRADES:
        raise ValueError('Expected an A/B/C/D/F threshold.')
    answerer = answerer or complete
    if record is None:
        record = {'responses': {}, 'calls': {}}
    responses = record['responses']
    if set(responses) - set(STAGES):
        raise ValueError('Unknown response stage in checkpoint.')

    def stage(name, prompt, evaluation=False):
        if name not in responses:
            result = (complete if evaluation else answerer)(prompt)
            if not isinstance(result['text'], str):
                raise ValueError('Completion must contain text.')
            text = (result['text'] if preserve_answer_whitespace and not evaluation
                    else result['text'].strip())
            if evaluation:
                text = ('Observed Information: ' + text).replace('\n\n', '\n')
                if read_belief(text) not in GRADES:
                    raise ValueError('Model returned an invalid belief grade; this stage was not saved. Resume to retry it.')
            responses[name] = text
            record['calls'][name] = {
                'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                'model': result.get('model'), 'usage': result.get('usage'),
                'finish_reason': result.get('finish_reason')}
            if record_prompts:
                record['calls'][name]['prompt'] = prompt
            for key in ('tool_trace', 'api_calls', 'steps', 'tool_calls'):
                if key in result:
                    record['calls'][name][key] = result[key]
            if save:
                save(record)
        if evaluation and read_belief(responses[name]) not in GRADES:
            raise ValueError('Checkpoint contains an invalid belief grade; use a new checkpoint or explicitly repair that stage.')
        return responses[name]

    initial = stage('initial_outputs', qa_prompt)
    evaluation = stage('initial_evaluations', evaluation_prompt(initial), True)
    if needs_refinement(read_belief(evaluation), threshold):
        revised = stage('refined_outputs', refinement_prompt(initial, evaluation))
        stage('refined_evaluations', evaluation_prompt(revised), True)
    return record
