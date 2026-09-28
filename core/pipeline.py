"""New inference orchestration using archived prompts and Algorithm 1.

Model response generation and answer-selection random streams are independent.
This runner was prepared in 2026; it is not a recovered 2024 executable.
"""
from types import SimpleNamespace

from . import base_prompt
from .io import load_json, write_json
from .selection import GRADES, rebuild_predictions
from .reasoning import STAGES, run_reasoning


def prompt_args(config):
    if config['prompt_format'] not in ('QCM-A', 'QCM-AE', 'QCM-ALE'):
        raise ValueError('Only the three published QCM formats are supported.')
    if config['threshold'] not in GRADES:
        raise ValueError('Invalid belief threshold.')
    return SimpleNamespace(model=config['prompt_model'], prompt_format=config['prompt_format'],
                           use_caption=config['use_caption'], options=config['options'])


def run_question(problems, qid, config, complete, record=None, save=None):
    """complete(prompt) returns text/model/usage/finish_reason; save checkpoints each call."""
    args = prompt_args(config)
    # Test annotations are never passed through to prompt construction. Training
    # examples retain their annotations as required for few-shot prompting.
    context = dict(problems)
    context[qid] = dict(problems[qid], answer=0, lecture='', solution='')
    shots = config['shot_qids']
    if qid in shots or any(problems[s]['split'] != 'train' for s in shots):
        raise ValueError('Few-shot examples must be training questions distinct from the test question.')
    return run_reasoning(
        base_prompt.build_prompt(context, shots, qid, args),
        lambda text: base_prompt.build_causal_reasoning_prompt(context, text, qid, args),
        lambda text, evaluation: base_prompt.build_refinement_prompt(context, text, shots, qid, evaluation, args),
        config['threshold'], complete, record, save)


def run_experiment(problems, qids, config, complete, checkpoint, input_hashes):
    """Run/resume a fixed ordered sample. Resume requires identical inputs and config."""
    qids = list(qids)
    if not qids or len(set(qids)) != len(qids):
        raise ValueError('Nonempty unique question IDs are required.')
    prompt_args(config)
    identity = {'config': config, 'qids': qids, 'input_sha256': input_hashes}
    if checkpoint.exists():
        state = load_json(checkpoint)
        if state.get('identity') != identity:
            raise ValueError('Checkpoint config, question order, data, captions or code differ.')
        if set(state['records']) - set(qids):
            raise ValueError('Checkpoint contains unexpected questions.')
    else:
        state = {'identity': identity, 'records': {}, 'complete': False}

    for qid in qids:
        def save(record):
            state['records'][qid] = record
            write_json(checkpoint, state)
        record = state['records'].get(qid)
        save(run_question(problems, qid, config, complete, record, save))

    mappings = {stage: {q: state['records'][q]['responses'][stage] for q in qids
                        if stage in state['records'][q]['responses']} for stage in STAGES}
    counts = {q: len(problems[q]['choices']) for q in qids}
    predictions, audits = {}, {}
    for model in ('GPT-3.5', 'CORE'):
        kwargs = {} if model == 'GPT-3.5' else {k: mappings[k] for k in STAGES[1:]}
        predictions[model], audits[model] = rebuild_predictions(
            qids, counts, mappings['initial_outputs'], seed=config['answer_selection_seed'],
            threshold=config['threshold'], **kwargs)
    state.update(predictions=predictions, audits=audits, complete=True)
    write_json(checkpoint, state)
    return state
