"""HotpotQA adapter for the shared CORE algorithm.

Question-only Standard/zero-shot CoT use pinned ReWOO instructions and parsers.
CORE templates are archived author code. This is an organized implementation,
not a recovered historical HotpotQA runner.

Answer metric functions are reorganized from the official HotpotQA definitions.
Original metric code: Copyright 2018 Zhilin Yang, Peng Qi, Saizheng Zhang,
Apache License 2.0 (see licenses/HotpotQA-CODE.txt). Modified here in 2026 to
remove nonstandard dependencies and expose answer-only aggregate metrics.
"""
from collections import Counter
import re
import string

from .faithful_reasoning_prompt import Causal_Reasoning_Belief_MODEL_PROMPT, Refinement_Model_Prompt
from .reasoning import run_reasoning
from .selection import needs_refinement, prefer_refinement, read_belief
from .hotpot_prompts import STANDARD_PROMPT, COT_PROMPT


def build_prompts(item, style='standard', context_mode='none'):
    if style not in ('standard', 'cot', 'react') or context_mode not in ('none', 'provided'):
        raise ValueError('Unsupported QA style or context mode.')
    task = 'Question: ' + item['question']
    if context_mode == 'provided':
        # All supplied paragraphs are used; supporting_facts never filter them.
        task += '\nContext:\n' + '\n'.join(title + ': ' + ' '.join(sentences)
                                           for title, sentences in item['context'])
    question = item['question'] if context_mode == 'none' else task
    # ReWOO IO/CoT: fixed instruction + empty fewshot newline + raw question.
    qa = ({'standard': STANDARD_PROMPT, 'cot': COT_PROMPT}[style] + '\n' + question + '\n'
          if style != 'react' else question)

    def evaluate(text):
        return Causal_Reasoning_Belief_MODEL_PROMPT.format(qa_response=task + '\nAnswer: ' + text)

    def refine(text, evaluation):
        # Change only the task-type wording from the archived choice-question
        # template. Keep the same QA format for the first and refined answers.
        template = Refinement_Model_Prompt.replace('scientific question-answering', 'question-answering')
        template = template.replace('correct option', 'correct answer')
        return template.format(qa_response=task + '\nAnswer: ' + text,
                               causal_reasoning='\n'.join(evaluation.split('\n')[:-1]),
                               qa_module_prompt=qa)
    return qa, evaluate, refine


def parse_answer(text, style='standard'):
    """ReWOO Evaluator parsing, including its original CoT prefix behavior."""
    if style == 'cot':
        return str(text).split('\n')[-1].replace('Answer:', '')
    if style in ('standard', 'react'):
        return str(text).strip('\n')
    raise ValueError('Unsupported QA style.')


def run_hotpot_question(item, config, complete, record=None, save=None, answerer=None):
    if config['style'] == 'react' and answerer is None:
        raise ValueError('ReAct requires a tool-enabled answer backend; it is not a plain prompt flag.')
    qa, evaluate, refine = build_prompts(item, config['style'], config['context_mode'])
    record = run_reasoning(qa, evaluate, refine, config['threshold'], complete, record, save,
                           answerer, preserve_answer_whitespace=True, record_prompts=True)
    responses = record['responses']
    initial = read_belief(responses['initial_evaluations'])
    refined_selected = (needs_refinement(initial, config['threshold']) and
                        prefer_refinement(initial, read_belief(responses['refined_evaluations'])))
    selected = responses['refined_outputs' if refined_selected else 'initial_outputs']
    record['initial_answer'] = parse_answer(responses['initial_outputs'], config['style'])
    record['selected_answer'] = parse_answer(selected, config['style'])
    record['parse_failed'] = not bool(record['selected_answer'].strip())
    record['refined_selected'] = refined_selected
    record['selected_stage'] = 'refined_outputs' if refined_selected else 'initial_outputs'
    for stage in ('initial_outputs', 'refined_outputs'):
        if stage in responses and config['style'] != 'react':
            record['calls'][stage]['steps'] = (1 if config['style'] == 'standard'
                                               else responses[stage].count('Step'))
    return record


def normalize_answer(text):
    """Official HotpotQA normalization: lowercase, punctuation, articles, whitespace."""
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_answers(predictions, gold):
    """Answer EM/F1 on a 0--100 scale. Model-judged Acc is computed separately."""
    if not gold or set(predictions) != set(gold):
        raise ValueError('Predictions must cover exactly the nonempty evaluation sample.')
    em, f1 = 0.0, 0.0
    for qid, reference in gold.items():
        pred, target = normalize_answer(predictions[qid]), normalize_answer(reference)
        em += float(pred == target)
        if pred != target and ({pred, target} & {'yes', 'no', 'noanswer'}):
            continue
        p, t = pred.split(), target.split()
        overlap = sum((Counter(p) & Counter(t)).values())
        if overlap:
            precision, recall = overlap/len(p), overlap/len(t)
            f1 += 2*precision*recall/(precision+recall)
    return {'EM': 100*em/len(gold), 'F1': 100*f1/len(gold)}
