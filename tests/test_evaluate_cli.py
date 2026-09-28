"""Damaged checkpoints must fail instead of reporting partial results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from core.io import sha256

ROOT = Path(__file__).resolve().parents[1]


class EvaluationCLITests(unittest.TestCase):
    def test_duplicate_ids_and_missing_methods_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            problems = {'q1': {'split': 'test', 'choices': ['x', 'y'], 'answer': 0},
                        'q2': {'split': 'test', 'choices': ['x', 'y'], 'answer': 1}}
            data = tmp/'problems.json'
            data.write_text(json.dumps(problems))
            for qids, predictions in [(['q1', 'q1', 'q2'], {'GPT-3.5': {'q1': 0, 'q2': 0}, 'CORE': {'q1': 0, 'q2': 0}}),
                                      (['q1', 'q2'], {}),
                                      (['q1', 'q2'], {'GPT-3.5': {'q1': 0, 'q2': 0}})]:
                with self.subTest(qids=qids, methods=list(predictions)):
                    run = tmp/'run.json'
                    run.write_text(json.dumps({'complete': True, 'identity': {'qids': qids,
                        'input_sha256': {'problems': sha256(data)}}, 'predictions': predictions}))
                    result = subprocess.run([sys.executable, str(ROOT/'evaluate_run.py'), str(run),
                        '--data-root', str(tmp)], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, '')
