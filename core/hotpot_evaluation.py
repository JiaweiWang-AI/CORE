"""ReWOO-style answer evaluation and explicit new-run resource accounting."""
from .hotpot import score_answers
from .hotpot_prompts import JUDGE_PROMPT


def grade_answer(question, prediction, gold, judge):
    prompt = JUDGE_PROMPT.format(query=question, result=prediction, answer=gold)
    response = judge(prompt)
    grade = response['text'].strip()
    metrics = score_answers({'q': prediction}, {'q': gold})
    return {'prediction': prediction, 'em': metrics['EM'] / 100, 'f1': metrics['F1'] / 100,
            'acc': int(grade == 'CORRECT'), 'grade': response['text'],
            'grade_format_valid': grade in ('CORRECT', 'INCORRECT'),
            'model': response.get('model'), 'usage': response.get('usage'),
            'finish_reason': response.get('finish_reason'), 'prompt': prompt}


def tokens(call):
    return (call.get('usage') or {}).get('total_tokens', 0)


def summarize(records, qids):
    metrics, failures = {}, {}
    for method in ('baseline', 'CORE'):
        totals = dict(EM=0, F1=0, Acc=0, AvgSteps=0, AvgTokens=0,
                      AvgAnswerStepsTotal=0, AvgAPICalls=0, AvgToolCalls=0)
        failures[method] = 0
        for q in qids:
            rec = records[q]
            grade = rec['evaluations'][method]
            for key, field in [('EM', 'em'), ('F1', 'f1'), ('Acc', 'acc')]:
                totals[key] += 100 * grade[field]
            failures[method] += not grade['grade_format_valid']
            stage = 'initial_outputs' if method == 'baseline' else rec['selected_stage']
            calls = ({'initial_outputs': rec['calls']['initial_outputs']}
                     if method == 'baseline' else rec['calls'])
            totals['AvgSteps'] += rec['calls'][stage]['steps']
            totals['AvgTokens'] += sum(tokens(c) for c in calls.values())
            totals['AvgAPICalls'] += sum(c.get('api_calls', 1) for c in calls.values())
            totals['AvgToolCalls'] += sum(c.get('tool_calls', 0) for c in calls.values())
            totals['AvgAnswerStepsTotal'] += sum(c.get('steps', 0) for c in calls.values())
        metrics[method] = {k: v / len(qids) for k, v in totals.items()}
    generation = sum(tokens(c) for q in qids for c in records[q]['calls'].values())
    grading = sum(tokens(g) for q in qids for g in records[q]['evaluations'].values())
    accounting = {'generation_total_tokens': generation, 'judge_total_tokens': grading,
                  'all_total_tokens': generation + grading,
                  'generation_api_calls': sum(c.get('api_calls', 1) for q in qids for c in records[q]['calls'].values()),
                  'judge_api_calls': sum(len(records[q]['evaluations']) for q in qids)}
    return metrics, failures, accounting
