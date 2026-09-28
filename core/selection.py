"""Reconstruction of CORE answer selection from saved generation transcripts.

Prepared 2026-09-27. This is not the original inference runner or an LLM rerun.
The parser and random valid-option fallback follow the saved CORE implementation
and ScienceQA. CORE selection follows Algorithm 1 (ties prefer the refinement).
"""
import random
import re

GRADES = ('A', 'B', 'C', 'D', 'F')
OPTIONS = ('A', 'B', 'C', 'D', 'E')


def needs_refinement(belief, threshold='A'):
    if belief not in GRADES or threshold not in GRADES:
        raise ValueError('Expected an A/B/C/D/F belief grade.')
    # A is highest confidence, so every grade is at or below threshold A.
    return GRADES.index(belief) >= GRADES.index(threshold)


def prefer_refinement(initial_belief, refined_belief):
    if initial_belief not in GRADES or refined_belief not in GRADES:
        raise ValueError('Expected an A/B/C/D/F belief grade.')
    return GRADES.index(refined_belief) <= GRADES.index(initial_belief)


def read_belief(text):
    matches = re.findall(r'\nBelief: ([A-Z])', text)
    return matches[-1] if matches else 'B'  # Retain the original parser default.


def read_answer(text):
    matches = re.findall(r'The answer is ([A-Z])', text)
    return matches[0] if len(matches) == 1 else 'FAILED'


def rebuild_predictions(qids, choice_counts, initial_outputs, *, seed=5,
                        initial_evaluations=None, refined_outputs=None,
                        refined_evaluations=None, threshold='A'):
    """Only response texts, option counts and ordered IDs enter this function.

    Gold labels and historical selected indices are deliberately not arguments.
    A fresh RNG belongs to this answer-selection pass, independently of shot
    sampling or API retry code. Valid responses consume no random draws.
    """
    qids = list(qids)
    if len(qids) != len(set(qids)):
        raise ValueError('Question IDs must be unique.')
    if threshold not in GRADES:
        raise ValueError('Expected an A/B/C/D/F threshold.')
    supplied = [x is not None for x in (initial_evaluations, refined_outputs, refined_evaluations)]
    core = all(supplied)
    if any(supplied) and not core:
        raise ValueError('CORE needs all three evaluation/refinement mappings.')
    rng = random.Random(seed)
    predictions, fallback_ids = {}, []
    refined_selected = 0
    for qid in qids:
        count = choice_counts[qid]
        if type(count) is not int or not 1 <= count <= len(OPTIONS):
            raise ValueError(f'Invalid option count for {qid}.')
        text = initial_outputs[qid]
        if core:
            belief = read_belief(initial_evaluations[qid])
            if needs_refinement(belief, threshold):
                revised_belief = read_belief(refined_evaluations[qid])
                if prefer_refinement(belief, revised_belief):
                    text = refined_outputs[qid]
                    refined_selected += 1
        answer = read_answer(text)
        if answer in OPTIONS[:count]:
            predictions[qid] = OPTIONS.index(answer)
        else:
            # Equivalent to ScienceQA get_pred_idx: sample only valid choices.
            predictions[qid] = rng.choice(range(count))
            fallback_ids.append(qid)
    return predictions, {'fallback_count':len(fallback_ids),
                         'fallback_ids':fallback_ids,
                         'refined_selected_count':refined_selected}
