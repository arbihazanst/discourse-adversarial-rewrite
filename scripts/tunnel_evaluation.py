"""Evaluate Gemma recipient warnings and frozen GAN text over the four SSH routes."""
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from time import monotonic

import adaptive_pipeline as core
from rewrite_validity import RUBRIC, VERSION, parameters
from service_pilot import endpoint, read_answer, service_connection, set_ports


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_archive(path, originals):
    archive = {}
    for row in core.read_csv(path):
        sid, step = row['sample_id'], int(row['step'])
        if (sid, step) in archive or step not in (1, 2) or not row['candidate'].strip():
            raise ValueError(f'Kandidat GAN duplikat atau tidak lengkap: {sid} step {step}')
        if sid not in originals or sha(originals[sid]) != row['original_sha256']:
            raise ValueError(f'Teks asli GAN tidak cocok: {sid}')
        archive[sid, step] = row
    for (sid, step), row in archive.items():
        if step == 2 and ((sid, 1) not in archive or sha(archive[sid, 1]['candidate']) != row['parent_sha256']):
            raise ValueError(f'Kandidat GAN step 2 tidak cocok dengan step 1: {sid}')
    return archive


class Requests:
    def __init__(self, db, routes, retry):
        self.db, self.routes, self.retry = db, routes, retry
        self.failed = set()
        db.execute('CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, answer TEXT, seconds REAL, error TEXT)')

    def ask(self, service, sid, stage, question, body):
        route = self.routes[service]
        payload = {'model': route['model'], **body}
        key = core.fingerprint([service, sid, stage, payload])
        saved = self.db.execute('SELECT answer, seconds, error FROM requests WHERE key=?', (key,)).fetchone()
        if saved and (not saved[2] or not self.retry or key in self.failed):
            return saved
        started = monotonic()
        try:
            response, _, _, _ = core.api.request_json(route['url'], '', payload, 120, 1)
            answer = read_answer(response, question, payload)
            if question == 'chat' and response['choices'][0].get('finish_reason') not in (None, 'stop'):
                raise ValueError('Jawaban Gemma terpotong.')
            if question == 'chat' and not answer.startswith('Peringatan: email ini ditandai sebagai spam.'):
                raise ValueError('Jawaban Gemma tidak diawali peringatan yang disyaratkan prompt.')
            error = ''
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError) as exc:
            answer, error = '', str(exc) or type(exc).__name__
            self.failed.add(key)
        result = answer, round(monotonic() - started, 4), error
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO requests VALUES (?,?,?,?)', (key, *result))
        return result

    def classify(self, defender, sid, stage, text):
        return self.ask(defender, sid, stage, 'spam', core.api.systemone_payload(text))

    def validate(self, sid, stage, original, candidate):
        checks, seconds = {}, 0.0
        for criterion in RUBRIC:
            params = parameters(original, candidate, criterion)
            question = {k: params[k] for k in ('criteria', 'instructions')}
            question['type'] = 'choice'
            body = {'state': params['state'], 'questions': {criterion: question}}
            label, elapsed, error = self.ask('pplx', sid, stage, criterion, body)
            seconds += elapsed
            if error:
                return 'ERROR', checks, seconds, error
            checks[criterion] = label
        labels = checks.values()
        verdict = 'INVALID' if 'FAIL' in labels else 'UNCERTAIN' if 'UNCERTAIN' in labels else 'VALID'
        return verdict, checks, seconds, ''


STEP_FIELDS = ['sample_id','defender','step','source','status','original_text','candidate','validation',
               *RUBRIC,'label','generation_sec','validation_sec','classification_sec','error']
BASE_FIELDS = ['sample_id','defender','true_label','label','seconds','error']


def evaluate_branch(api, config, archive, prompts, branch, sid, original, defender):
    previous = ''
    for step in (1, 2):
        stage = f'{branch}{step}'
        r = dict.fromkeys(STEP_FIELDS, '')
        r.update(sample_id=sid, defender=defender, step=step, original_text=original)
        if branch == 'gan':
            entry = archive.get((sid, step))
            if entry is None:
                r['status'] = 'NO_ARCHIVE'
                yield r
                return
            candidate = entry['candidate']
            r['source'] = f"{entry['source_csv']}#{entry['source_record']}"
        else:
            prompt = (prompts['ATTACKER_PROMPT_1'] if step == 1 else
                      prompts['ATTACKER_PROMPT_2_TEMPLATE'].format(original_email=original, first_modified=previous))
            body = core.generation_payload({**config['attacker'], 'transport':'openai',
                'model':api.routes['gemma']['model']}, prompt, previous or original)
            candidate, r['generation_sec'], r['error'] = api.ask('gemma', sid, stage, 'chat', body)
            r['source'] = 'gemma_spam_warning'
        r['candidate'] = candidate
        if r['error']:
            r['status'] = 'ERROR'
        elif ' '.join(candidate.split()) in {' '.join(original.split()), ' '.join(previous.split())}:
            r['status'] = r['validation'] = 'NO_CHANGE'
        else:
            verdict, checks, r['validation_sec'], r['error'] = api.validate(sid, stage, original, candidate)
            r.update(checks)
            r['status'] = r['validation'] = verdict
            if verdict == 'VALID':
                r['label'], r['classification_sec'], r['error'] = api.classify(defender, sid, stage, candidate)
                r['status'] = 'ERROR' if r['error'] else 'CLASSIFIED'
        yield r
        if r['status'] != 'CLASSIFIED' or r['label'] != 'SPAM':
            return
        previous = candidate


def run(config, mode, task, model='all', overrides=(), dry=False, check=False, retry=False, run_id=None,
        dataset=None, output=None, limit=0):
    rows = core.api.load_dataset(core.rooted(dataset or config['dataset']), {'clean'})
    originals = {r['sample_id']: r['email_text'] for r in rows}
    if not rows or len(originals) != len(rows):
        raise ValueError('Dataset harus berisi original dengan sample_id unik.')
    if mode == 'pilot':
        spam = [r for r in rows if r['true_label'] == 'SPAM'][:5]
        ham = [r for r in rows if r['true_label'] == 'NOT_SPAM'][:5]
        rows = spam + ham
        if len(rows) != 10:
            raise ValueError('Pilot membutuhkan 5 SPAM dan 5 NOT_SPAM.')
    elif limit:
        rows = rows[:limit]
    defenders = ['clef','d3'] if model == 'all' else [model]
    if set(defenders) - {'clef','d3'}:
        raise ValueError('Defender tunnel: clef, d3 atau all. PPLX bertugas sebagai evaluator.')
    branches = {'baseline':(), 'llm':('gemma',), 'gan':('gan',), 'all':('gemma','gan')}[task]
    required = set(defenders)
    if branches: required.add('pplx')
    if 'gemma' in branches: required.add('gemma')
    routes = {k:dict(v) for k,v in config['service_pilot'].items()}
    set_ports(routes, overrides)
    if required - routes.keys():
        raise ValueError(f'Profil tidak memuat layanan {sorted(required - routes.keys())}')
    routes = {k:v for k,v in routes.items() if k in required}
    archive = load_archive(core.rooted(config['gan_archive']), originals) if 'gan' in branches else {}
    prompts = json.loads((core.ROOT/'decision_models/paper_prompts.json').read_text(encoding='utf-8')) if 'gemma' in branches else {}
    manifest = dict(protocol='warning_and_gan_replay_v1', task=task, routes=routes, mode=mode, run_id=run_id,
        sample_hash=core.fingerprint([(r['sample_id'],r['true_label'],sha(r['email_text'])) for r in rows]),
        prompts=prompts, gan_archive=core.fingerprint(list(archive.values())), rubric=VERSION,
        criteria=RUBRIC, validator_parameters=parameters('', '', 'purpose'), classifier=core.api.SYSTEMONE_QUESTION,
        generation={k:config['attacker'][k] for k in ('temperature','max_tokens')} if prompts else {})
    base = core.ROOT/('experimenting/outputs' if config['profile']=='experimenting' else 'outputs')
    folder = (core.rooted(output) if output else base/'tunnel'/mode/task) / core.fingerprint(manifest)[:16]
    if folder.is_relative_to((core.ROOT/'experimenting').resolve()) != (config['profile']=='experimenting'):
        raise ValueError('Folder hasil tidak sesuai dengan profil.')
    for name,route in routes.items():
        print(f"{name}: {route['model']} | {endpoint(route)}")
    print(f'{len(rows)} original | {task} | {folder}', flush=True)
    if 'gan' in branches:
        print('Arsip GAN untuk pilihan ini: ' + ', '.join(f"step {i} = {sum((r['sample_id'],i) in archive for r in rows)}" for i in (1,2)))
    if dry:
        return 0
    if check:
        with service_connection(routes):
            for name, route in routes.items():
                if name == 'gemma':
                    q, body = 'chat', {'messages':[{'role':'user','content':'Reply with OK.'}], 'max_tokens':16,'stream':False}
                else:
                    q, body = 'spam', core.api.systemone_payload('Team meeting tomorrow.')
                payload = {'model':route['model'], **body}
                response, _, _, _ = core.api.request_json(route['url'], '', payload, 120, 1)
                read_answer(response, q, payload)
        return 0
    with core.file_lock(folder/'run.lock'), service_connection(routes), closing(sqlite3.connect(folder/'checkpoint.sqlite')) as db:
        core.atomic_json(folder/'run.json', manifest)
        requests = Requests(db, routes, retry)
        baseline, outputs = [], {(b,i):[] for b in branches for i in (1,2)}
        start = monotonic()
        for n,row in enumerate(rows,1):
            sid,truth,original = row['sample_id'],row['true_label'],row['email_text']
            for defender in defenders:
                label,elapsed,error = requests.classify(defender,sid,'baseline',original)
                baseline.append(dict(sample_id=sid,defender=defender,true_label=truth,label=label,seconds=elapsed,error=error))
                if truth != 'SPAM' or label != 'SPAM':
                    continue
                for b in branches:
                    for rec in evaluate_branch(requests, config, archive, prompts, b, sid, original, defender):
                        step = rec['step']
                        outputs[b,step].append(rec)
                        if rec['candidate']:
                            core.atomic_csv(folder/'rewrites'/b/defender/f'step{step}'/(sha(sid)[:16]+'.csv'), [rec], STEP_FIELDS)
            if n==1 or n%25==0 or n==len(rows):
                print(core.api.progress(n,len(rows),monotonic()-start,0),flush=True)
        core.atomic_csv(folder/'baseline.csv', baseline, BASE_FIELDS)
        summary = {'baseline':{}, 'steps':{}}
        for d in defenders:
            subset = [r for r in baseline if r['defender']==d]
            counts = Counter(('TP' if r['label']=='SPAM' else 'TN') if r['label']==r['true_label'] else
                             ('FP' if r['label']=='SPAM' else 'FN') for r in subset if not r['error'])
            summary['baseline'][d] = {**{k:counts[k] for k in ('TP','TN','FP','FN')}, 'errors':sum(bool(r['error']) for r in subset)}
            base_stats = summary['baseline'][d]
            base_stats.update(core.paper_evaluation.classification_metrics(counts))
            base_stats['accuracy'] = core.paper_evaluation.divide(counts['TP']+counts['TN'], sum(counts.values()))
            base_stats['spam_recall'] = core.paper_evaluation.divide(counts['TP'], counts['TP']+counts['FN'])
            base_stats['macro_f1'] = core.paper_evaluation.divide(
                base_stats['spam_f1']+base_stats['ham_f1'],2) if None not in (base_stats['spam_f1'],base_stats['ham_f1']) else None
        for (b,i), records in outputs.items():
            core.atomic_csv(folder/f'{b}_step{i}.csv', records, STEP_FIELDS)
            for d in defenders:
                sub = [r for r in records if r['defender']==d]
                summary['steps'][f'{b}_{i}_{d}'] = dict(status=dict(Counter(r['status'] for r in sub)),
                    validity=dict(Counter(r['validation'] for r in sub if r['validation'])),
                    validated_label_flips=sum(r['label']=='NOT_SPAM' for r in sub))
        core.atomic_json(folder/'ringkasan.json', summary)
        errors = any(r['error'] for r in baseline) or any(r['error'] for sub in outputs.values() for r in sub)
    print(f"Hasil: {folder / 'ringkasan.json'}",flush=True)
    return 2 if errors else 0

