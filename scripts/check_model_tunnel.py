"""Check four forwarded model services with a refund and a RAG question."""
from contextlib import contextmanager
import socket
import subprocess
import time

from run_decision_benchmark import request_json

MODELS = {
    11438: 'perplexity-ai/pplx-decider-flash-v1.2',
    11439: 'clef-flash-q4',
    11440: 'd3-flash-q4',
    11441: 'google/gemma-3-27b-it',
}
DEPARTMENTS = {'billing': 'Payments and refunds', 'technical': 'Technical issues', 'sales': 'Product purchases'}


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.35):
            return True
    except OSError:
        return False


@contextmanager
def tunnel():
    missing = [port for port in MODELS if not listening(port)]
    if not missing:
        yield
        return
    command = ['ssh', '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']
    for port in missing:
        command += ['-L', f'127.0.0.1:{port}:127.0.0.1:{port}']
    command += ['-p', '28659', 'rinaldi@37.60.243.124']
    print('Membuka tunnel SSH. Masukkan password di prompt OpenSSH jika diminta.', flush=True)
    proc = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 120
        while True:
            if proc.poll() is not None:
                raise OSError(f'SSH berhenti dengan kode {proc.returncode}. Periksa pesan OpenSSH.')
            if all(listening(port) for port in missing):
                break
            if time.monotonic() >= deadline:
                raise OSError('Waktu login SSH habis.')
            time.sleep(0.25)
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def check(dry=False):
    failed = 0
    def url(port):
        path = '/v1/chat/completions' if port == 11441 else '/v1/systemone'
        return f'http://127.0.0.1:{port}{path}'
    for port, model in MODELS.items():
        print(f'{model}  {url(port)}')
    if dry:
        return 0
    with tunnel():
        for port, model in MODELS.items():
            chat = port == 11441
            if chat:
                body = dict(model=model,messages=[dict(role='user',content='Explain Retrieval-Augmented Generation in two sentences.')],
                            temperature=0.2,max_tokens=150,stream=False)
            else:
                body = dict(model=model,state='The customer requests a refund for a duplicate payment.',
                            questions={'department': dict(type='choice',instructions='Select the responsible department.',criteria=DEPARTMENTS)})
            try:
                response, _, ms, _ = request_json(url(port), '', body, 120 if chat else 60, 1)
                if response.get('model') not in (None, model):
                    raise ValueError('ID model respons berbeda dari request.')
                if chat:
                    answer = response['choices'][0]['message']['content']
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError('Respons chat kosong.')
                else:
                    answer = response['answers']['department']['choice']
                    if answer not in DEPARTMENTS:
                        raise ValueError('Pilihan department tidak dikenali.')
                print(f'OK {model} ({ms/1000:.2f} s)  {answer}',flush=True)
            except (OSError,RuntimeError,ValueError,TypeError,KeyError,IndexError,AttributeError) as exc:
                print(f'GAGAL {model}: {exc}',flush=True)
                failed += 1
    print(f'{len(MODELS)-failed}/{len(MODELS)} layanan merespons dengan format yang diharapkan.')
    return 2 if failed else 0
