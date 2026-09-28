"""Configuration precedence at the public CLI; no model or network calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_hotpotqa


class ConfigurationTests(unittest.TestCase):
    def invoke(self, overrides):
        config = json.loads((run_hotpotqa.ROOT/'configs/hotpotqa.json').read_text())
        config.update(style='cot', threshold='C')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps(config))
            (root/'data.json').write_text(json.dumps([
                {'_id': 'q', 'question': 'Capital of France?', 'answer': 'Paris'}]))
            argv = ['run_hotpotqa.py', '--config', str(root/'config.json'),
                    '--data', str(root/'data.json'), '--output', str(root/'result.json'),
                    '--dry-run', *overrides]
            output = io.StringIO()
            with patch('sys.argv', argv), contextlib.redirect_stdout(output):
                run_hotpotqa.main()
            self.assertFalse((root/'result.json').exists())
            return output.getvalue()

    def test_configured_style_and_threshold_survive_omitted_cli_flags(self):
        output = self.invoke([])
        self.assertIn("['cot']", output)
        self.assertIn('tau=C', output)

    def test_explicit_cli_style_and_threshold_override_config(self):
        output = self.invoke(['--style', 'all', '--threshold', 'A'])
        self.assertIn("['standard', 'cot', 'react']", output)
        self.assertIn('tau=A', output)


if __name__ == '__main__':
    unittest.main()
