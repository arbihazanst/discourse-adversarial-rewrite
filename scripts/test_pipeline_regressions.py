"""Regression checks for run isolation, provenance, failures, ETA and safe merges."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import merge_decision_csvs as merge
import run_decision_benchmark as runner
import summarize_decision_results as summary

HERE = Path(__file__).parent


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'inputs.csv'
        self.output = self.root / 'predictions.csv'
        with self.dataset.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['input_id', 'sample_id', 'kind', 'true_label', 'email_text'])
            writer.writeheader()
            writer.writerows([
                dict(input_id='A:clean', sample_id='A', kind='clean', true_label='SPAM', email_text='prize'),
                dict(input_id='B:clean', sample_id='B', kind='clean', true_label='NOT_SPAM', email_text='meeting'),
                dict(input_id='A:attack', sample_id='A', kind='frozen_primary_step1', true_label='SPAM', email_text='rewrite'),
            ])
        self.args = ['--dataset', str(self.dataset), '--out', str(self.output), '--model', 'clef',
                     '--transport', 'direct', '--checkpoint-id', 'revision-one-Q8']

    def run_mock(self, extra=(), failure=False):
        log = io.StringIO()
        with patch.object(runner, 'request_json') as request, contextlib.redirect_stdout(log):
            if failure:
                request.side_effect = RuntimeError('simulated backend failure')
            else:
                request.return_value = ({'label': 'SPAM'}, 200, 10, 'test-hash')
            code = runner.main(self.args + list(extra))
        return code, request.call_count, log.getvalue()

    def test_resume_requires_same_endpoint_checkpoint_and_dataset(self):
        self.assertEqual(self.run_mock()[:2], (0, 3))
        self.assertEqual(self.run_mock()[:2], (0, 0))
        before = self.output.read_bytes()
        for change in (['--direct-host', 'another-host'], ['--checkpoint-id', 'revision-two-Q8']):
            with self.subTest(change=change), self.assertRaisesRegex(SystemExit, 'different'):
                self.run_mock(change)
        self.dataset.write_bytes(self.dataset.read_bytes() + b'\n')
        with self.assertRaisesRegex(SystemExit, 'different'):
            self.run_mock()
        self.assertEqual(before, self.output.read_bytes())

    def test_pilot_or_limit_can_resume_into_full_run(self):
        self.assertEqual(self.run_mock(['--limit', '1'])[:2], (0, 1))
        code, calls, log = self.run_mock()
        self.assertEqual((code, calls), (0, 2))
        self.assertIn('attempted=2/2', log)
        self.assertIn('ETA=00:00:00', log)

    def test_all_errors_resume_and_retry(self):
        self.assertEqual(self.run_mock(failure=True)[:2], (2, 3))
        code, calls, log = self.run_mock()
        self.assertEqual((code, calls), (2, 0))
        self.assertIn('unresolved_skipped=3', log)
        self.assertEqual(self.run_mock(['--retry-errors'])[:2], (0, 3))
        self.assertEqual(self.run_mock()[:2], (0, 0))

    def test_all_errors_summary_has_zero_counts_and_na_rates(self):
        self.run_mock(failure=True)
        stats = self.root / 'stats.csv'
        result = subprocess.run([sys.executable, str(HERE / 'summarize_decision_results.py'),
                                 '--dataset', str(self.dataset), '--predictions', str(self.output),
                                 '--out', str(stats)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with stats.open(newline='') as f:
            row = next(csv.DictReader(f))
        for key in ('clean_valid', 'attack_valid', 'observed_flips', 'initially_spam_with_valid_attack'):
            self.assertEqual(row[key], '0')
        self.assertEqual(row['clean_accuracy'], 'NA')
        self.assertEqual(row['label_flip_rate'], 'NA')

    def test_merge_and_summary_keep_distinct_runs(self):
        self.run_mock()
        other = self.root / 'revision-two.csv'
        self.run_mock(['--out', str(other), '--checkpoint-id', 'revision-two-Q8'])
        combined = self.root / 'combined.csv'
        with contextlib.redirect_stdout(io.StringIO()):
            merge.main(['--inputs', str(self.output), str(other), '--out', str(combined)])
        predictions = summary.load(combined, hashlib.sha256(self.dataset.read_bytes()).hexdigest())
        self.assertEqual(len(predictions), 6)
        self.assertEqual(len({key[0] for key in predictions}), 2)
        with self.assertRaisesRegex(SystemExit, 'different dataset'):
            summary.load(combined, 'wrong-hash')

    def test_merge_preserves_files_on_alias_and_invalid_input(self):
        self.run_mock()
        original = self.output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            merge.main(['--inputs', str(self.output), '--out', str(self.output)])
        self.assertEqual(self.output.read_bytes(), original)
        invalid = self.root / 'invalid.csv'
        invalid.write_text('bad,header\n1,2\n')
        dest = self.root / 'merged.csv'
        dest.write_bytes(b'preserve existing output')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            merge.main(['--inputs', str(self.output), str(invalid), '--out', str(dest)])
        self.assertEqual(dest.read_bytes(), b'preserve existing output')

    def test_reject_legacy_output_without_mutation(self):
        self.output.write_text('defender,input_id,status\nclef,A:clean,OK\n')
        original = self.output.read_bytes()
        with self.assertRaisesRegex(SystemExit, 'legacy'):
            self.run_mock()
        self.assertEqual(self.output.read_bytes(), original)

    def test_tool_provenance_must_match(self):
        for source in ({'type': 'document', 'id': 'pplx'}, {'type': 'tool', 'id': 'wrong'}, {}):
            with self.subTest(source=source), self.assertRaises(ValueError):
                runner.extract({'sources': [{**source, 'content': {'label': 'SPAM'}}]}, 'tool', 'pplx')
        response = {'sources': [{'source': {'type': 'tool', 'id': 'pplx'},
                                 'document': ['{"label":"NOT_SPAM"}']}]}
        self.assertEqual(runner.extract(response, 'tool', 'pplx')[0], 'NOT_SPAM')
        with self.assertRaises(ValueError):
            runner.extract(response, 'tool')

    def test_eta_includes_attempt_time_delay_and_pending_only(self):
        self.assertIn('ETA=estimating', runner.progress(0, 5, 0, 1))
        self.assertIn('ETA=00:00:36', runner.progress(2, 5, 20, 2))
        self.assertIn('elapsed=00:00:22', runner.progress(2, 5, 20, 2))
        self.assertIn('ETA=00:00:00', runner.progress(5, 5, 50, 2))

    def test_raw_responses_do_not_overwrite_previous_attempts(self):
        raw = self.root / 'raw'
        # A valid JSON response with no decision is saved even when parsing fails.
        with patch.object(runner, 'request_json', return_value=({}, 200, 1, 'hash')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(self.args + ['--raw-dir', str(raw), '--limit', '1']), 2)
        self.run_mock(['--raw-dir', str(raw), '--limit', '1', '--retry-errors'])
        self.assertEqual(len(list(raw.glob('*.json'))), 2)
        with self.output.open(newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len({row['raw_file'] for row in rows}), 2)
        for row in rows:
            self.assertEqual(hashlib.sha256(Path(row['raw_file']).read_bytes()).hexdigest(), row['raw_saved_sha256'])

    def checkpoint(self):
        return self.output.with_name(self.output.name + '.checkpoint.json')

    def test_interrupt_then_resume_only_unfinished_records(self):
        response = ({'label': 'SPAM'}, 200, 10, 'test-hash')
        with patch.object(runner, 'request_json', side_effect=[response, KeyboardInterrupt]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(self.args), 130)
        checkpoint = json.loads(self.checkpoint().read_text())
        self.assertEqual(checkpoint['state'], 'interrupted')
        self.assertEqual(checkpoint['successful_input_ids'], ['A:clean'])
        self.assertEqual(checkpoint['pending_count'], 2)
        self.assertEqual(self.run_mock(['--resume'])[:2], (0, 2))
        final = json.loads(self.checkpoint().read_text())
        self.assertEqual((final['state'], final['successful_count'], final['pending_count']), ('complete', 3, 0))
        with self.output.open(newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row['input_id'] for row in rows}), 3)

    def test_missing_corrupt_or_stale_checkpoint_is_rebuilt_from_csv(self):
        self.run_mock(['--limit', '1'])
        # A checkpoint that incorrectly claims everything is complete must not skip work.
        self.checkpoint().write_text('{"state":"complete","successful_count":9999}')
        self.assertEqual(self.run_mock(['--resume'])[:2], (0, 2))
        self.checkpoint().unlink()
        self.assertEqual(self.run_mock(['--resume'])[:2], (0, 0))
        self.assertEqual(json.loads(self.checkpoint().read_text())['successful_count'], 3)
        self.checkpoint().write_text('{broken')
        self.assertEqual(self.run_mock(['--resume'])[:2], (0, 0))
        self.assertEqual(json.loads(self.checkpoint().read_text())['state'], 'complete')

    def test_checkpoint_does_not_hide_failed_results(self):
        self.run_mock(failure=True)
        checkpoint = json.loads(self.checkpoint().read_text())
        self.assertEqual((checkpoint['state'], checkpoint['failed_count']), ('incomplete', 3))
        self.assertEqual(self.run_mock(['--resume'])[:2], (2, 0))
        self.assertEqual(self.run_mock(['--resume', '--retry-errors'])[:2], (0, 3))
        self.assertEqual(json.loads(self.checkpoint().read_text())['failed_count'], 0)

    def test_atomic_checkpoint_failure_preserves_previous_snapshot(self):
        self.run_mock(['--limit', '1'])
        path = self.checkpoint()
        before = path.read_bytes()
        with patch.object(runner.os, 'replace', side_effect=OSError('simulated disk error')):
            with self.assertRaises(OSError):
                runner.save_checkpoint(path, self.output, 'test', {}, [], {}, 'running')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_resume_supports_results_from_previous_runner(self):
        self.run_mock()
        with self.output.open(newline='') as file:
            rows = list(csv.DictReader(file))
        identity = {key: rows[0][key] for key in ('defender', 'backend_mode', 'model_id', 'endpoint',
                                                'checkpoint_id', 'tool_id', 'dataset_sha256')}
        config = {**identity, 'template': None, 'question': runner.SYSTEMONE_QUESTION,
                  'runner_sha256': runner.COMPATIBLE_RUNNER_HASHES[0]}
        previous_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        with self.output.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=runner.COLUMNS)
            writer.writeheader()
            writer.writerows({**row, 'run_id': previous_id} for row in rows)
        self.assertEqual(self.run_mock(['--resume'])[:2], (0, 0))
        self.assertEqual(json.loads(self.checkpoint().read_text())['run_id'], previous_id)
        with self.assertRaisesRegex(SystemExit, 'different'):
            self.run_mock(['--resume', '--checkpoint-id', 'different-weights'])

    def test_dry_run_does_not_create_checkpoint(self):
        self.assertEqual(self.run_mock(['--resume', '--dry-run'])[:2], (0, 0))
        self.assertFalse(self.output.exists())
        self.assertFalse(self.checkpoint().exists())


if __name__ == '__main__':
    unittest.main()
