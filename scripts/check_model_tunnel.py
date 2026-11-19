"""Manage local model connections and test them using harmless example inputs."""
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.parse import urlsplit

from run_decision_benchmark import request_json

DEPARTMENTS = {'billing':'Payments and refunds','technical':'Technical issues','sales':'Product purchases'}


def default_profile():
    root = Path(__file__).resolve().parent.parent
    local = root/'experimenting/adaptive_local.json'
    return json.loads((local if local.exists() else root/'experimenting/adaptive_config.json').read_text(encoding='utf-8-sig'))


def routes(config):
    return {'pplx':config['rewrite_evaluator'], **config['defenders'], 'gemma':config['attacker']}


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1',port), timeout=0.35):
            return True
    except OSError:
        return False


@contextmanager
def tunnel(config=None):
    config = default_profile() if config is None else config
    if config['profile'] == 'main':
        yield None
        return
    ssh = config.get('ssh')
    if not ssh:
        yield None
        return
    forwards = {}
    for name, route in routes(config).items():
        parsed = urlsplit(route['url'])
        if parsed.hostname != '127.0.0.1' or parsed.port is None:
            raise ValueError('Profil SSH membutuhkan URL 127.0.0.1 dengan port lokal eksplisit.')
        remote = int(ssh['remote_ports'][name])
        if not 1 <= remote <= 65535:
            raise ValueError('Port remote harus 1Ã¢â‚¬â€œ65535.')
        if parsed.port in forwards and forwards[parsed.port] != remote:
            raise ValueError('Satu port lokal dipetakan ke dua layanan berbeda.')
        forwards[parsed.port] = remote
    missing = {local:remote for local,remote in forwards.items() if not listening(local)}
    if not missing:
        yield None
        return
    destination = ssh['destination']
    if not isinstance(destination,str) or destination.startswith('-') or any(c.isspace() for c in destination):
        raise ValueError('Tujuan SSH harus berbentuk user@host.')
    port = int(ssh['port'])
    if not 1 <= port <= 65535:
        raise ValueError('Port SSH harus 1Ã¢â‚¬â€œ65535.')
    command = ['ssh','-N','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=15',
               '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3']
    for local, remote in missing.items():
        command += ['-L',f'127.0.0.1:{local}:127.0.0.1:{remote}']
    command += ['-p',str(port),destination]
    print('Membuka tunnel experimenting. Masukkan password di prompt OpenSSH jika diminta.',flush=True)
    proc = subprocess.Popen(command)
    try:
        deadline = time.monotonic()+120
        while True:
            if proc.poll() is not None:
                raise OSError(f'SSH berhenti dengan kode {proc.returncode}.')
            if all(listening(local) for local in missing):
                break
            if time.monotonic() >= deadline:
                raise OSError('Waktu login SSH habis.')
            time.sleep(0.25)
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def check(dry=False, config=None):
    config = default_profile() if config is None else config
    for name, route in routes(config).items():
        print(f"{name}: {route['model']} | {route['url']}")
    if dry:
        return 0
    failed = 0
    with tunnel(config):
        for name, route in routes(config).items():
            chat = name == 'gemma'
            if chat:
                body = dict(model=route['model'],messages=[dict(role='user',content='Explain Retrieval-Augmented Generation in two sentences.')],
                            temperature=0.2,max_tokens=150,stream=False)
            else:
                body = dict(model=route['model'],state='The customer requests a refund for a duplicate payment.',
                            questions={'department':dict(type='choice',instructions='Select the responsible department.',criteria=DEPARTMENTS)})
            try:
                import adaptive_pipeline as core
                response, _, ms, _ = request_json(route['url'],core.credentials(route),body,120 if chat else 60,1)
                if response.get('model') not in (None,route['model']):
                    raise ValueError('ID model respons berbeda dari request.')
                if chat:
                    answer = response['choices'][0]['message']['content']
                    if not isinstance(answer,str) or not answer.strip():
                        raise ValueError('Respons chat kosong.')
                else:
                    answer = response['answers']['department']['choice']
                    if answer not in DEPARTMENTS:
                        raise ValueError('Pilihan department tidak dikenali.')
                print(f'OK {name} ({ms/1000:.2f} s) {answer}',flush=True)
            except (OSError,RuntimeError,ValueError,TypeError,KeyError,IndexError,AttributeError) as exc:
                print(f'GAGAL {name}: {exc}',flush=True)
                failed += 1
    return 2 if failed else 0
