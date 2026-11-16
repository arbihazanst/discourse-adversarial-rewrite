"""Pilot of four local services using original emails and previously recorded text pairs."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.parse import urlsplit, urlunsplit

import adaptive_pipeline as core
from rewrite_validity import RUBRIC, VERSION, parameters


def endpoint(route, port=None):
    parsed = urlsplit(route['url'])
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Gunakan URL HTTP(S) model tanpa kredensial.')
    if port is None:
        return route['url']
    if not 1 <= port <= 65535:
        raise ValueError('Port harus 1–65535.')
    host = f'[{parsed.hostname}]' if ':' in parsed.hostname else parsed.hostname
    return urlunsplit(parsed._replace(netloc=f'{host}:{port}'))


def set_ports(routes, overrides):
    for override in overrides:
        name, sep, value = override.partition('=')
        if not sep or name not in routes or not value.isascii() or not value.isdecimal():
            raise ValueError('Format --port: pplx=11438, clef=11439, d3=11440 atau gemma=11441 (satu per opsi).')
        routes[name]['url'] = endpoint(routes[name], int(value))


REMOTE_PORTS = {'pplx': 11438, 'clef': 11439, 'd3': 11440, 'gemma': 11441}


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.4):
            return True
    except OSError:
        return False


@contextmanager
def service_connection(routes):
    forwards = []
    for name, route in routes.items():
        url = urlsplit(endpoint(route))
        if url.hostname != '127.0.0.1':
            continue
        port = url.port or (443 if url.scheme == 'https' else 80)
        if not listening(port):
            forwards.append((port, REMOTE_PORTS[name]))
    if not forwards:
        yield
        return
    args = ['ssh', '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=12',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']
    for local, remote in forwards:
        args.extend(['-L', f'127.0.0.1:{local}:127.0.0.1:{remote}'])
    args.extend(['-p', '28659', 'rinaldi@37.60.243.124'])
    print('Membuka tunnel. Masukkan password di prompt OpenSSH jika diminta.', flush=True)
    proc = subprocess.Popen(args)
    try:
        deadline = time.monotonic() + 120
        while True:
            if proc.poll() is not None:
                raise OSError(f'SSH berhenti (exit {proc.returncode}). Periksa pesan OpenSSH di atas.')
            if all(listening(local) for local, _ in forwards):
                break
            if time.monotonic() >= deadline:
                raise OSError('Waktu tunggu login SSH habis.')
            time.sleep(0.25)
        yield
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def tasks(config, routes, mode='pilot'):
    dataset = core.rooted(config['dataset'])
    rows = core.api.load_dataset(dataset, {'clean'})
    sample = rows
    if mode == 'pilot':
        sample = []
        for label in ('SPAM', 'NOT_SPAM'):
            group = [r for r in rows if r['true_label'] == label][:5]
            if len(group) != 5:
                raise ValueError(f'Dataset membutuhkan lima original {label}.')
            sample.extend(group)
    bank = {r['sample_id']: r for r in core.read_csv(core.rooted(config['step1_bank']))}
    for row in sample:
        for name in ('clef', 'd3'):
            body = {'model': routes[name]['model'], **core.api.systemone_payload(row['email_text'])}
            yield name, row['sample_id'], 'spam', row['true_label'], body
    for row in sample:
        pair = bank.get(row['sample_id'])
        if not pair or not pair['rewrite_text'].strip():
            continue
        if hashlib.sha256(row['email_text'].encode()).hexdigest() != pair['original_text_sha256']:
            raise ValueError('Hash original berbeda dari bank rewrite: ' + row['sample_id'])
        for criterion in RUBRIC:
            args = parameters(row['email_text'], pair['rewrite_text'], criterion)
            body = {'model': routes['pplx']['model'], 'state': args['state'],
                    'questions': {criterion: {'type': 'choice', 'instructions': args['instructions'], 'criteria': args['criteria']}}}
            yield 'pplx', row['sample_id'], criterion, '', body
    yield 'gemma', 'smoke', 'chat', '', {'model': routes['gemma']['model'],
        'messages': [{'role': 'user', 'content': 'Explain Retrieval-Augmented Generation in two sentences.'}],
        'temperature': 0.2, 'max_tokens': 150, 'stream': False}


def read_answer(response, question, body):
    if response.get('model') not in (None, body['model']):
        raise ValueError('ID model pada respons berbeda dari request.')
    if question == 'chat':
        answer = response['choices'][0]['message']['content']
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Gemma mengembalikan teks kosong.')
        return answer.strip()
    answer = response['answers'][question]['choice']
    if answer not in body['questions'][question]['criteria']:
        raise ValueError(f'Pilihan tidak dikenal untuk {question}.')
    return answer


def run(config, overrides=(), dry=False, retry=False, mode='pilot', run_id=None):
    routes = {name: dict(route) for name, route in config['service_pilot'].items()}
    set_ports(routes, overrides)
    plan = list(tasks(config, routes, mode))
    identity = core.fingerprint([routes, VERSION, plan])
    prefix = Path('experimenting/outputs') if config['profile'] == 'experimenting' else Path('outputs')
    if run_id:
        identity = core.fingerprint([identity, run_id])
    folder = core.rooted(prefix / ('pilot_layanan10' if mode == 'pilot' else 'layanan_full') / identity[:12])
    for name, route in routes.items():
        print(f"{name}: {route['model']}  {endpoint(route)}")
    print(f'{len(plan)} pemeriksaan | {folder}', flush=True)
    if dry:
        return 0
    result_file = folder / 'hasil.csv'
    fields = ['service', 'sample_id', 'question', 'truth', 'answer', 'seconds', 'status', 'error']
    with core.file_lock(folder / 'run.lock'), service_connection(routes):
        manifest = folder / 'run.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['identity'] != identity:
            raise ValueError('Identitas checkpoint berbeda.')
        core.atomic_json(manifest, dict(identity=identity, profile=config['profile'], services=routes, rubric=VERSION))
        results = {(r['service'], r['sample_id'], r['question']): r for r in core.read_csv(result_file)} if result_file.exists() else {}
        for service, sid, question, truth, body in plan:
            key = service, sid, question
            old = results.get(key)
            if old and (old['status'] == 'OK' or not retry):
                continue
            row = dict(service=service, sample_id=sid, question=question, truth=truth, answer='', seconds='', status='ERROR', error='')
            try:
                response, _, elapsed_ms, _ = core.api.request_json(endpoint(routes[service]), '', body, 120, 1)
                row['seconds'] = round(elapsed_ms / 1000, 3)
                row['answer'] = read_answer(response, question, body)
                row['status'] = 'OK'
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
                row['error'] = str(exc)
            results[key] = row
            core.atomic_csv(result_file, results.values(), fields)
            print(f"{service} {sid} {question}: {row['status']}", flush=True)
        summary = {'checks': len(plan), 'ok': sum(r['status'] == 'OK' for r in results.values()), 'rubric': VERSION}
        for name in ('clef', 'd3'):
            valid = [r for r in results.values() if r['service'] == name and r['status'] == 'OK']
            summary[name] = {'classified': len(valid), 'correct': sum(r['truth'] == r['answer'] for r in valid)}
        verdicts = {}
        for sid in dict.fromkeys(s for name, s, _, _, _ in plan if name == 'pplx'):
            checks = [results['pplx', sid, q] for q in RUBRIC]
            labels = {r['answer'] for r in checks}
            if any(r['status'] != 'OK' for r in checks):
                verdicts[sid] = 'ERROR'
            else:
                verdicts[sid] = 'INVALID' if 'FAIL' in labels else 'UNCERTAIN' if 'UNCERTAIN' in labels else 'VALID'
        summary['pplx_saved_pairs'] = verdicts
        core.atomic_json(folder / 'ringkasan.json', summary)
        print(f"{summary['ok']}/{len(plan)} pemeriksaan berhasil | {folder / 'ringkasan.json'}")
    return 0 if summary['ok'] == len(plan) else 2
