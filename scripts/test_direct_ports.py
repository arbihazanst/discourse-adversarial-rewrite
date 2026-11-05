#!/usr/bin/env python3
"""Offline tests: port registry, native endpoint routing, and native label parsing."""
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from run_decision_benchmark import DIRECT_HOST_PORTS, systemone_payload

HERE = Path(__file__).parent


class FakeDecisionBackend(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.__class__.requests.append((self.path, body, self.headers.get('Authorization')))
        if self.path == '/v1/systemone':
            self.assert_http_body = body
            label = 'SPAM' if 'prize' in body['state'] else 'NOT_SPAM'
            reply = {'answers': {'spam': {'choice': label,
                'probabilities': {'SPAM': .88 if label == 'SPAM' else .12,
                                  'NOT_SPAM': .12 if label == 'SPAM' else .88}}}}
        elif self.path == '/actual-pplx-native-route':
            label = 'SPAM' if 'prize' in body['email'] else 'NOT_SPAM'
            reply = {'result': {'label': label}}
        else:
            self.send_response(404); self.end_headers(); return
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class TestDirectPorts(unittest.TestCase):
    def test_real_host_port_contract(self):
        self.assertEqual(DIRECT_HOST_PORTS, {'pplx': 30002, 'clef': 30005, 'd3': 30006})
        self.assertEqual(systemone_payload('Hi')['state'], 'Hi')
        self.assertEqual(set(systemone_payload('Hi')['questions']['spam']['criteria']), {'SPAM', 'NOT_SPAM'})
        config = json.loads((HERE.parent/'decision_models/decision_model_config.json').read_text())
        self.assertEqual({m['model_key']: int(m['host_port']) for m in config['models']}, DIRECT_HOST_PORTS)

    def test_endpoint_defaults_visible_on_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self.make_dataset(Path(tmp))
            for m in ('clef', 'd3'):
                result = subprocess.run([sys.executable, str(HERE/'run_decision_benchmark.py'),
                    '--dataset', str(dataset), '--model', m, '--transport', 'direct', '--dry-run'],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'127.0.0.1:{DIRECT_HOST_PORTS[m]}/v1/systemone', result.stdout)

    @staticmethod
    def make_dataset(where):
        path = where/'benchmark.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['input_id', 'sample_id', 'kind', 'true_label', 'email_text', 'source_attack_model'])
            w.writeheader()
            w.writerows([
                dict(input_id='A:clean', sample_id='A', kind='clean', true_label='SPAM', email_text='prize draw', source_attack_model=''),
                dict(input_id='B:clean', sample_id='B', kind='clean', true_label='NOT_SPAM', email_text='meeting reminder', source_attack_model=''),
            ])
        return path

    def test_native_http_request_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = self.make_dataset(root)
            server = HTTPServer(('127.0.0.1', 0), FakeDecisionBackend)
            FakeDecisionBackend.requests = []
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for model in ('clef', 'd3'):
                    dest = root/f'{model}.csv'
                    cmd = [sys.executable, str(HERE/'run_decision_benchmark.py'),
                           '--dataset', str(dataset), '--model', model, '--transport', 'direct',
                           '--direct-base-url', f'http://127.0.0.1:{server.server_port}',
                           '--out', str(dest), '--attempts', '1']
                    run = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
                    again = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertEqual(again.returncode, 0, again.stderr)
                    with dest.open(newline='') as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(len(rows), 2)
                    self.assertEqual({r['predicted_label'] for r in rows}, {'SPAM','NOT_SPAM'})
                    self.assertTrue(all(r['backend_mode'] == 'direct' for r in rows))
                self.assertEqual(len(FakeDecisionBackend.requests), 4)
                self.assertTrue(all(route == '/v1/systemone' for route, _, _ in FakeDecisionBackend.requests))
                self.assertTrue(all('spam' in body['questions'] for _, body, _ in FakeDecisionBackend.requests))
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)

    def test_pplx_requires_verified_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = self.make_dataset(root)
            cmd = [sys.executable, str(HERE/'run_decision_benchmark.py'),
                   '--dataset', str(dataset), '--model', 'pplx', '--transport', 'direct', '--dry-run']
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn('requires --pplx-native-path', res.stderr)

    def test_pplx_explicit_verified_template(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dataset = self.make_dataset(root)
            template = root/'pplx-template.json'
            template.write_text('{"email": "__EMAIL_TEXT__"}')
            server = HTTPServer(('127.0.0.1', 0), FakeDecisionBackend)
            FakeDecisionBackend.requests = []
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                cmd = [sys.executable, str(HERE/'run_decision_benchmark.py'), '--dataset', str(dataset),
                       '--model', 'pplx', '--transport', 'direct',
                       '--direct-base-url', f'http://127.0.0.1:{server.server_port}',
                       '--pplx-native-path', '/actual-pplx-native-route',
                       '--pplx-native-template', str(template), '--out', str(root/'pplx.csv')]
                res = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, res.stderr+res.stdout)
                self.assertEqual(len(FakeDecisionBackend.requests), 2)
                self.assertEqual(FakeDecisionBackend.requests[0][0], '/actual-pplx-native-route')
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
