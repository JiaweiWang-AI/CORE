"""ScienceQA subgroup definitions; derived from the official evaluator."""
COLUMNS = ('NAT', 'SOC', 'LAN', 'TXT', 'IMG', 'NO', 'G1-6', 'G7-12', 'Avg')


def compute_scores(problems, predictions):
    """Compare zero-based option indices using the official ScienceQA subgroup definitions."""
    test = {qid: p for qid, p in problems.items() if p['split'] == 'test'}
    if not test or set(test) != set(predictions):
        raise ValueError('Predictions must cover exactly the complete, nonempty test split.')
    groups = {name: [] for name in COLUMNS}
    for qid, p in test.items():
        pred = predictions[qid]
        if type(pred) is not int or not 0 <= pred < len(p['choices']):
            raise ValueError(f'Invalid zero-based prediction for question {qid}.')
        subject = {'natural science': 'NAT', 'social science': 'SOC',
                   'language science': 'LAN'}[p['subject']]
        grade = int(p['grade'].removeprefix('grade'))
        if not 1 <= grade <= 12:
            raise ValueError(f'Invalid grade for question {qid}.')
        memberships = [subject, 'G1-6' if grade <= 6 else 'G7-12', 'Avg']
        if p['hint']:
            memberships.append('TXT')
        if p['image']:
            memberships.append('IMG')
        if not p['hint'] and not p['image']:
            memberships.append('NO')
        for group in memberships:
            groups[group].append(pred == p['answer'])
    return {group: f'{100 * sum(correct) / len(correct):.2f}' if correct else '-1'
            for group, correct in groups.items()}

