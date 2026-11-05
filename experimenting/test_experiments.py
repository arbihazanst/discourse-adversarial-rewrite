"""Offline tests: mock Ollama, resume, run isolation, and Windows batch launchers."""
import contextlib
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import classify
import pipeline

HERE = Path(__file__).resolve().parent


class FakeOllama(BaseHTTPRequestHandler):
    requests = []
    digests = {}
    fail_model = None

    def log_message(self, *args):
        pass

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/api/tags':
            self.send_error(404)
            return
        self.send_json({'models': [{'name': tag, 'digest': digest} for tag, digest in self.digests.items()]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests.append((self.path, body))
        if self.path != '/v1/systemone':
            self.send_error(404)
            return
        if body['model'] == self.fail_model:
            self.send_json({'model': body['model'], 'response': 'This is spam'})
            return
        label = 'SPAM' if body['state'] == 'prize' else 'NOT_SPAM'
        self.send_json({'model': body['model'], 'answers': {'spam': {
            'type': 'choice', 'choice': label,
            'probabilities': {'SPAM': .9 if label == 'SPAM' else .1,
                              'NOT_SPAM': .1 if label == 'SPAM' else .9}}}})


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test experiment ', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'input.csv'
        with self.dataset.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['input_id', 'sample_id', 'kind', 'true_label', 'email_text'])
            writer.writeheader()
            writer.writerows([
                dict(input_id='1:clean', sample_id='1', kind='clean', true_label='SPAM', email_text='prize'),
                dict(input_id='2:clean', sample_id='2', kind='clean', true_label='NOT_SPAM', email_text='meeting'),
                dict(input_id='1:attack', sample_id='1', kind='frozen_primary_step1', true_label='SPAM', email_text='rewrite'),
            ])
        self.output = self.root / 'outputs'
        self.args = ['--dataset', str(self.dataset), '--output-dir', str(self.output), '--attempts', '1']
        self.installed = {tag: {'digest': tag + '-digest'} for tag in classify.MODELS.values()}

    def start_server(self):
        FakeOllama.requests = []
        FakeOllama.digests = {tag: tag + '-digest' for tag in classify.MODELS.values()}
        FakeOllama.fail_model = None
        server = HTTPServer(('127.0.0.1', 0), FakeOllama)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.addCleanup(close)
        return f'http://127.0.0.1:{server.server_port}'

    def test_offline_defaults_same_dataset_and_isolated_outputs(self):
        args = classify.parser().parse_args(['--model', 'pplx', '--dry-run'])
        self.assertEqual(args.dataset, HERE.parent / 'dataset/paper_baseline3000.csv')
        self.assertEqual(args.base_url, 'http://127.0.0.1:11435')
        self.assertEqual(args.output_dir, HERE / 'outputs' / 'paper3000_baseline')
        with patch.object(classify, 'installed_models') as server, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(self.args + ['--dry-run']), 0)
            server.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_end_to_end_two_models_resume_and_metrics(self):
        url = self.start_server()
        args = self.args + ['--base-url', url]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(args), 0)
            self.assertEqual(pipeline.main(args + ['--resume']), 0)
        self.assertEqual(len(FakeOllama.requests), 6)
        self.assertEqual([body['model'] for _, body in FakeOllama.requests],
                         ['server:pplxdeciderflash:latest'] * 3 + ['clef:27b'] * 3)
        for route, payload in FakeOllama.requests:
            self.assertEqual(route, '/v1/systemone')
            self.assertEqual(set(payload), {'model', 'state', 'questions', 'keep_alive'})
            self.assertEqual(set(payload['questions']['spam']['criteria']), {'SPAM', 'NOT_SPAM'})
        with (self.output / 'metrics.csv').open(newline='') as file:
            metrics = list(csv.DictReader(file))
        self.assertEqual(len(metrics), 2)
        self.assertTrue(all(row['clean_accuracy'] == '100.00%' and row['label_flip_rate'] == '100.00%' for row in metrics))
        for model in classify.MODELS:
            snapshot = json.loads((self.output / (model + '_ollama.csv.checkpoint.json')).read_text())
            self.assertEqual(snapshot['state'], 'complete')
            self.assertEqual(snapshot['successful_count'], 3)
        FakeOllama.digests['server:pplxdeciderflash:latest'] = 'replacement-weights'
        with self.assertRaisesRegex(SystemExit, 'different'), contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(args + ['--resume'])
        self.assertEqual(len(FakeOllama.requests), 6)

    def test_failure_stops_pipeline_and_retry_completes(self):
        url = self.start_server()
        args = self.args + ['--base-url', url]
        FakeOllama.fail_model = 'server:pplxdeciderflash:latest'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(args), 2)
        self.assertEqual(len(FakeOllama.requests), 3)
        self.assertFalse((self.output / 'clef_ollama.csv').exists())
        self.assertFalse((self.output / 'metrics.csv').exists())
        FakeOllama.fail_model = None
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(args + ['--resume', '--retry-errors']), 0)
        self.assertEqual(len(FakeOllama.requests), 9)

    def test_preflight_checks_both_models_before_inference(self):
        installed = {'server:pplxdeciderflash:latest': {'digest': 'test'}}
        with patch.object(classify, 'installed_models', return_value=installed), patch.object(classify, 'request_json') as request:
            with self.assertRaisesRegex(SystemExit, 'clef:27b'), contextlib.redirect_stdout(io.StringIO()):
                pipeline.main(self.args)
            request.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_interrupt_and_resume(self):
        response = ({'model': 'clef:27b', 'answers': {'spam': {'choice': 'SPAM'}}}, 200, 1, 'hash')
        args = self.args + ['--model', 'clef']
        with patch.object(classify, 'installed_models', return_value=self.installed), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(classify, 'request_json', side_effect=[response, KeyboardInterrupt]):
                self.assertEqual(classify.main(args), 130)
            with patch.object(classify, 'request_json', return_value=response) as request:
                self.assertEqual(classify.main(args + ['--resume']), 0)
                self.assertEqual(request.call_count, 2)

    def test_output_must_stay_in_experimenting(self):
        with self.assertRaisesRegex(SystemExit, 'inside experimenting'):
            pipeline.main(self.args + ['--output-dir', str(HERE.parent / 'outputs'), '--dry-run'])

    @unittest.skipUnless(os.name == 'nt', 'Windows batch test')
    def test_batch_launchers_from_other_directory_with_spaces(self):
        env = dict(os.environ, EXPERIMENT_PYTHON=sys.executable)
        for name in ('run_ollama_pplx.bat', 'run_ollama_clef.bat', 'run_ollama_all.bat'):
            command = f'call "{HERE / name}" --dry-run --pilot --output-dir "{self.output}"'
            result = subprocess.run('cmd.exe /d /s /c "' + command + '"', cwd=self.root,
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('selected=10', result.stdout)
            self.assertIn('127.0.0.1:11435/v1/systemone', result.stdout)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows batch test')
    def test_batch_forwards_resume_and_exit_status(self):
        url = self.start_server()
        env = dict(os.environ, EXPERIMENT_PYTHON=sys.executable)
        FakeOllama.fail_model = 'server:pplxdeciderflash:latest'
        command = (f'call "{HERE / "run_ollama_all.bat"}" --dataset "{self.dataset}" '
                   f'--output-dir "{self.output}" --base-url {url} --attempts 1 --resume')
        result = subprocess.run('cmd.exe /d /s /c "' + command + '"', cwd=self.root,
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse((self.output / 'clef_ollama.csv').exists())


if __name__ == '__main__':
    unittest.main()
