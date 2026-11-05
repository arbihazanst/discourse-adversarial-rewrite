#!/usr/bin/env python3
"""Classify the shared CEAS dataset through local Ollama /v1/systemone."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import uuid

from benchmark_common import (COLUMNS, LABELS, SYSTEMONE_QUESTION, choose_decision,
                              existing_results, load_dataset, progress, request_json,
                              save_checkpoint, utc_now)

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / 'dataset/paper_baseline3000.csv'
MODELS = {'pplx': 'server:pplxdeciderflash:latest', 'clef': 'clef:27b'}
DEFAULT_BASE_URL = 'http://127.0.0.1:11435'
INFERENCE_VERSION = 1


def parser(single=True):
    ap = argparse.ArgumentParser(description=__doc__)
    if single:
        ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--dataset', type=Path, default=DATASET)
    ap.add_argument('--base-url', default=os.environ.get('EXPERIMENT_OLLAMA_URL', DEFAULT_BASE_URL))
    ap.add_argument('--output-dir', type=Path, default=HERE / 'outputs' / 'paper3000_baseline',
                    help='Must stay inside experimenting; relative paths are relative to experimenting')
    ap.add_argument('--pilot', action='store_true', help='5 spam + 5 ham; also 5 rewrites if using the old transfer dataset')
    ap.add_argument('--limit', type=int, default=0, help='0=all selected records')
    ap.add_argument('--resume', action='store_true', help='Resume saved results (also the default)')
    ap.add_argument('--retry-errors', action='store_true')
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--attempts', type=int, default=3)
    ap.add_argument('--delay', type=float, default=0)
    ap.add_argument('--keep-alive', default='5m')
    ap.add_argument('--dry-run', action='store_true', help='Offline input/route check; no files or API calls')
    ap.add_argument('--check-server', action='store_true', help='Check installed model digests and resume compatibility; no inference')
    return ap


def prepare(args):
    if args.limit < 0 or args.timeout <= 0 or args.attempts < 1 or not math.isfinite(args.delay) or args.delay < 0:
        raise SystemExit('Require limit/delay >= 0, timeout > 0, attempts >= 1')
    if args.pilot and args.limit:
        raise SystemExit('Choose --pilot or --limit, not both')
    output_dir = args.output_dir if args.output_dir.is_absolute() else HERE / args.output_dir
    args.output_dir = output_dir.resolve()
    try:
        args.output_dir.relative_to(HERE)
    except ValueError:
        raise SystemExit('--output-dir must stay inside experimenting')
    args.base_url = args.base_url.rstrip('/')
    if not args.base_url.startswith(('http://', 'https://')):
        raise SystemExit('--base-url must be an HTTP(S) URL')
    rows = load_dataset(args.dataset, {'clean', 'frozen_primary_step1'})
    if not rows:
        raise SystemExit('The dataset contains no benchmark inputs')
    if args.pilot:
        selected = []
        for kind, label in (('clean', 'SPAM'), ('clean', 'NOT_SPAM'), ('frozen_primary_step1', 'SPAM')):
            selected.extend([r for r in rows if r['kind'] == kind and r['true_label'] == label][:5])
    else:
        selected = rows[:args.limit] if args.limit else rows
    print(f'Dataset: {args.dataset} | total={len(rows)} selected={len(selected)}')
    print(f'Endpoint: {args.base_url}/v1/systemone | outputs: {args.output_dir}')
    return selected


def installed_models(base_url, timeout):
    try:
        with urlopen(base_url + '/api/tags', timeout=min(timeout, 30)) as response:
            data = json.load(response)
    except (OSError, URLError, ValueError) as exc:
        raise SystemExit(f'Cannot read Ollama models at {base_url}/api/tags: {exc}')
    if not isinstance(data, dict) or not isinstance(data.get('models'), list):
        raise SystemExit('Unexpected Ollama /api/tags response')
    return {m.get('name') or m.get('model'): m for m in data['models'] if isinstance(m, dict)}


def plan_run(args, model, installed):
    tag = MODELS[model]
    digest = installed.get(tag, {}).get('digest')
    if not isinstance(digest, str) or not digest.strip():
        raise SystemExit(f'Ollama model {tag} is missing or has no digest. Install that exact tag on {args.base_url}.')
    identity = {
        'defender': model, 'backend_mode': 'ollama_systemone', 'model_id': tag,
        'endpoint': args.base_url + '/v1/systemone', 'checkpoint_id': digest,
        'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(), 'tool_id': '',
    }
    config = {**identity, 'questions': SYSTEMONE_QUESTION, 'inference_version': INFERENCE_VERSION}
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = args.output_dir / f'{model}_ollama.csv'
    prior = existing_results(output, run_id)
    print(f'{model}: {tag} | digest={digest} | existing_inputs={len(prior)}')
    return identity, run_id, output, prior


def classify(args, items, plan):
    identity, run_id, output_path, prior = plan
    pending, skip_ok, skip_error = [], 0, 0
    for item in items:
        old = prior.get(item['input_id'])
        if old and old.get('status') == 'OK' and old.get('predicted_label') in LABELS:
            skip_ok += 1
        elif old and not args.retry_errors:
            skip_error += 1
        else:
            pending.append(item)
    checkpoint = output_path.with_name(output_path.name + '.checkpoint.json')
    raw_dir = args.output_dir / ('raw_' + identity['defender'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint, output_path, run_id, identity, items, prior, 'running')
    print(f"{identity['model_id']}: completed={skip_ok} pending={len(pending)} unresolved_skipped={skip_error}")
    print(progress(0, len(pending), 0, args.delay) if pending else 'ETA=00:00:00')
    ok, errors, measured = 0, 0, 0.0
    exists = output_path.exists() and output_path.stat().st_size > 0
    try:
        with output_path.open('a', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            if not exists:
                writer.writeheader()
                file.flush()
                os.fsync(file.fileno())
            for index, item in enumerate(pending, 1):
                started = time.perf_counter()
                result = dict.fromkeys(COLUMNS, '')
                result.update(identity, run_id=run_id, timestamp_utc=utc_now(),
                              input_id=item['input_id'], sample_id=item['sample_id'], kind=item['kind'],
                              true_label=item['true_label'], source_attack_model=item.get('source_attack_model', ''))
                payload = {'model': identity['model_id'], 'state': item['email_text'],
                           'questions': SYSTEMONE_QUESTION, 'keep_alive': args.keep_alive}
                try:
                    response, status, elapsed, raw_hash = request_json(identity['endpoint'], '', payload, args.timeout, args.attempts)
                    result.update(http_status=status, latency_ms=elapsed, raw_sha256=raw_hash)
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = raw_dir / (run_id[:12] + '_' + uuid.uuid4().hex + '.json')
                    saved = json.dumps(response, ensure_ascii=False, indent=2).encode('utf-8')
                    raw_path.write_bytes(saved)
                    result.update(raw_file=str(raw_path), raw_saved_sha256=hashlib.sha256(saved).hexdigest())
                    if not isinstance(response, dict) or response.get('model') != identity['model_id']:
                        raise ValueError('Ollama response model does not match the requested tag')
                    answer = response.get('answers', {}).get('spam')
                    if not isinstance(answer, dict) or answer.get('choice') not in LABELS:
                        raise ValueError('Missing structured answers.spam.choice in System One response')
                    label, p_spam, p_not_spam = choose_decision(answer)
                    result.update(predicted_label=label, p_spam=p_spam, p_not_spam=p_not_spam, status='OK')
                    ok += 1
                except Exception as exc:
                    result.update(status='ERROR', error=(str(exc) or repr(exc))[:700])
                    errors += 1
                writer.writerow(result)
                file.flush()
                os.fsync(file.fileno())
                prior[item['input_id']] = result
                save_checkpoint(checkpoint, output_path, run_id, identity, items, prior, 'running', item['input_id'])
                measured += time.perf_counter() - started
                print(f"{identity['defender']} {item['input_id']} {result['status']} {result['predicted_label']} "
                      f"{result['error'][:100]} | " + progress(index, len(pending), measured, args.delay), flush=True)
                if args.delay and index < len(pending):
                    time.sleep(args.delay)
    except KeyboardInterrupt:
        save_checkpoint(checkpoint, output_path, run_id, identity, items, prior, 'interrupted')
        print('\nInterrupted; rerun this launcher with --resume to continue.')
        return 130
    save_checkpoint(checkpoint, output_path, run_id, identity, items, prior,
                    'incomplete' if errors or skip_error else 'complete')
    print(f'Finished: success={ok} errors={errors} already_complete={skip_ok} unresolved_skipped={skip_error}')
    if skip_error:
        print('Use --resume --retry-errors to retry failed records.')
    return 2 if errors or skip_error else 0


def execute(args, models, analyze=False):
    items = prepare(args)
    if args.dry_run:
        for model in models:
            print(f'DRY RUN: {MODELS[model]} -> {args.output_dir / (model + "_ollama.csv")}')
        return 0
    installed = installed_models(args.base_url, args.timeout)
    # Validate every selected model and existing log before any inference.
    plans = [plan_run(args, model, installed) for model in models]
    if args.check_server:
        print('Server/model/resume checks passed. No inference performed.')
        return 0
    for plan in plans:
        code = classify(args, items, plan)
        if code:
            return code
    if analyze:
        from merge_results import main as merge
        from summarize_results import main as summarize
        combined = args.output_dir / 'all_predictions.csv'
        merge(['--inputs', *[str(plan[2]) for plan in plans], '--out', str(combined)])
        summarize(['--dataset', str(args.dataset), '--predictions', str(combined),
                   '--out', str(args.output_dir / 'metrics.csv')])
    return 0


def install_stop_handler():
    def stop_requested(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_requested)


def main(argv=None):
    args = parser().parse_args(argv)
    return execute(args, [args.model])


if __name__ == '__main__':
    install_stop_handler()
    sys.exit(main())
