"""Resumable baseline -> rewrite1 -> classify1 -> rewrite2 -> classify2 experiment.

Shared controller for main and experimenting profiles. All inference is explicit;
--dry-run is offline. The old static benchmark remains separately available.
"""
import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from urllib.request import urlopen
import uuid

import run_decision_benchmark as api
import paper_evaluation

ROOT = Path(__file__).resolve().parent.parent
STAGES = ('baseline', 'rewrite1', 'classify1', 'rewrite2', 'classify2')
FIELDS = ['run_id','sample_id','stage','status','label','p_spam','p_not_spam',
          'artifact','artifact_sha256','request_sha256','source','latency_ms','error']
REWRITE_FIELDS = ['defender','sample_id','step','true_label','original_text','first_rewrite_text',
                  'rewrite_text','baseline_label','generation_status','classification_status','predicted_label',
                  'generation_time_sec','classification_time_sec','source','generation_error','classification_error']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def atomic_replace(source,target):
    # Windows scanners/readers can briefly deny replacement of a closed file.
    # Retain atomic replacement; never delete the last valid checkpoint first.
    for attempt in range(6):
        try:
            os.replace(source,target)
            return
        except PermissionError as exc:
            if os.name!='nt' or getattr(exc,'winerror',None) not in (5,32,33) or attempt==5:raise
            time.sleep(0.05*(2**attempt))


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
        atomic_replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader(); writer.writerows(rows); f.flush(); os.fsync(f.fileno())
        atomic_replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_model_key(name):
    if not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name) or name in {'all','generation_cache','evaluation_cache'}:
        raise ValueError('Model keys must be safe lowercase names, e.g. clef or my_model; all is reserved')


def rewrite_record(ledger, row, step):
    sid=row['sample_id']; generated=ledger.get(sid,'rewrite'+str(step)) or {}
    classified=ledger.get(sid,'classify'+str(step)) or {}
    baseline=ledger.ok(sid,'baseline') or {}
    seconds=lambda event: round(float(event['latency_ms'])/1000,6) if event.get('latency_ms') not in ('',None) else ''
    return dict(defender=ledger.folder.name,sample_id=sid,step=step,true_label=row['true_label'],
        original_text=row['email_text'],
        first_rewrite_text=ledger.text(sid,'rewrite1') if step==2 and ledger.ok(sid,'rewrite1') else '',
        rewrite_text=ledger.text(sid,'rewrite'+str(step)) if generated.get('status')=='OK' else '',
        baseline_label=baseline.get('label',''),generation_status=generated.get('status','PENDING'),
        classification_status=classified.get('status','PENDING'),predicted_label=classified.get('label',''),
        generation_time_sec=seconds(generated),classification_time_sec=seconds(classified),
        source=generated.get('source',''),generation_error=generated.get('error',''),classification_error=classified.get('error',''))


def save_individual_rewrite(ledger, row, step):
    if not ledger.get(row['sample_id'],'rewrite'+str(step)):
        return
    safe_id=re.sub(r'[^a-zA-Z0-9_-]', '_', row['sample_id'])[:60]
    suffix=hashlib.sha256(row['sample_id'].encode()).hexdigest()[:16]
    record=rewrite_record(ledger,row,step)
    atomic_csv(ledger.folder/'rewrites'/('step'+str(step))/(safe_id+'_'+suffix+'.csv'),[record],REWRITE_FIELDS)


def export_rewrites(ledger, rows):
    for step in (1,2):
        records=[]
        for row in rows:
            sid=row['sample_id'];base=ledger.ok(sid,'baseline')
            if row['true_label']!='SPAM' or not base or base['label']!='SPAM':continue
            first=ledger.ok(sid,'classify1')
            if step==2 and (not first or first['label']!='SPAM'):continue
            records.append(rewrite_record(ledger,row,step))
            save_individual_rewrite(ledger,row,step)
        atomic_csv(ledger.folder/('step'+str(step)+'_rewrites.csv'),records,REWRITE_FIELDS)


@contextmanager
def file_lock(path):
    """OS-released lock, including after a crash; no stale PID-file cleanup."""
    path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b')
    try:
        f.seek(0,2)
        if f.tell()==0: f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f'Another process is using {path.parent}') from exc
        yield
    finally:
        f.close()


def rooted(value):
    p=Path(value)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def credentials(config):
    name=config.get('key_env','')
    key=os.environ.get(name,'') if name else ''
    if name and not key:
        raise ValueError(f'Set environment variable {name}')
    return key


def selected_models(config, requested):
    names=config.get('active_models',list(config['defenders'])) if requested=='all' else [requested]
    if not isinstance(names,list) or not names or len(set(names))!=len(names):
        raise ValueError('active_models must be a nonempty list of unique defender keys')
    for name in names:
        validate_model_key(name)
        if name not in config['defenders']:raise ValueError('Unknown defender: '+name)
    return names


def validate_endpoint(config,attacker=False,live=False,timeout=30):
    allowed={'openai','ollama_chat'} if attacker else {'openwebui','ollama','direct','tool'}
    if config.get('transport') not in allowed:
        raise ValueError(f'Unsupported transport {config.get("transport")}')
    if not config.get('url','').startswith(('http://','https://')):
        raise ValueError('Endpoint must be an HTTP(S) URL')
    if config['transport']=='tool' and (not config.get('tool_id') or not config.get('model')):
        raise ValueError('Tool route requires verified tool_id and orchestration model')
    if config['transport']=='tool':
        # Invalidate old tool checkpoints that did not actually request execution.
        config['tool_protocol']='legacy_fixed_spam_v2'
    if not live:
        return
    if not config.get('model'):
        raise ValueError('Set the verified model/Pipe ID; PPLX Pipe is not assumed to exist')
    if config['transport'] in {'ollama','ollama_chat'}:
        from urllib.parse import urlsplit
        parsed=urlsplit(config['url']); url=f'{parsed.scheme}://{parsed.netloc}/api/tags'
        with urlopen(url,timeout=timeout) as f: tags=json.load(f)
        models={m.get('name') or m.get('model'):m for m in tags.get('models',[])}
        revision=models.get(config['model'],{}).get('digest')
        if not revision: raise ValueError(f'Model {config["model"]} not installed at {url}')
        if config.get('revision') and config['revision']!=revision:
            raise ValueError(f'Configured digest differs for {config["model"]}')
        config['revision']=revision
    if not config.get('revision'):
        raise ValueError(f'Set actual revision/checkpoint for {config["model"]}; it is part of resume identity')
    credentials(config)


def defender_payload(config,text):
    transport=config['transport']
    if transport=='tool':return api.tool_payload(config['model'],config['tool_id'],text)
    if transport=='ollama':
        return {'model':config['model'],'state':text,'questions':api.SYSTEMONE_QUESTION,'keep_alive':'5m'}
    if transport=='direct':
        if config.get('template') is not None:
            if '__EMAIL_TEXT__' not in json.dumps(config['template']):
                raise ValueError('Native template must contain __EMAIL_TEXT__')
            return api.apply_text_marker(config['template'],text)
        return api.systemone_payload(text)
    payload={'model':config['model'],'messages':[{'role':'user','content':text}],'stream':False}
    return payload


def parse_decision(config,response,text=None):
    if config['transport']=='ollama':
        if response.get('model')!=config['model']:
            raise ValueError('Response model does not match requested Ollama tag')
        answer=response.get('answers',{}).get('spam',{})
        if answer.get('choice') not in api.LABELS:
            raise ValueError('Missing structured answers.spam.choice')
        return api.choose_decision(answer)
    if config['transport']=='direct':return api.choose_decision(response)
    if config['transport']=='tool' and config.get('tool_protocol')=='legacy_fixed_spam_v2' and text is None:
        raise ValueError('Expected email is required to verify fixed PPLX tool parameters')
    expected=text if config.get('tool_protocol')=='legacy_fixed_spam_v2' else None
    return api.extract(response,'tool' if config['transport']=='tool' else 'pipe',config.get('tool_id'),expected)


def generation_payload(config,prompt,text):
    messages=[{'role':'system','content':prompt},{'role':'user','content':text}]
    if config['transport']=='ollama_chat':
        return {'model':config['model'],'messages':messages,'stream':False,
                'options':{'temperature':config['temperature'],'num_predict':config['max_tokens']}}
    return {'model':config['model'],'messages':messages,'temperature':config['temperature'],
            'max_tokens':config['max_tokens'],'stream':False}


def validate_rewrite(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('Empty rewrite')
    if re.match(r"\s*(i cannot|i can't|i am unable|i'm unable|i'm sorry|i am sorry)\b",text,re.I):
        raise ValueError('Attacker returned an apparent refusal')
    return text.strip()


def parse_generation(config,response):
    if config['transport']=='ollama_chat':
        if response.get('model')!=config['model']:raise ValueError('Wrong attacker model')
        if response.get('done') is not True or response.get('done_reason')=='length':
            raise ValueError('Incomplete/truncated attacker response')
        return validate_rewrite(response.get('message',{}).get('content'))
    choice=response.get('choices',[{}])[0]
    if choice.get('finish_reason') not in (None,'stop'):
        raise ValueError(f'Unusable attacker finish_reason: {choice.get("finish_reason")}')
    return validate_rewrite(choice.get('message',{}).get('content'))


class Ledger:
    def __init__(self,root,model,identities,valid_ids,stages=STAGES):
        self.root=root; self.folder=root/model; self.ids=identities; self.valid_ids=valid_ids
        self.path=self.folder/'events.csv'; self.events={}
        if self.path.exists():
            with self.path.open(encoding='utf-8',newline='') as f:
                reader=csv.DictReader(f)
                if reader.fieldnames!=FIELDS:raise ValueError('Incompatible event schema')
                for r in reader:
                    if None in r or any(v is None for v in r.values()):raise ValueError('Truncated event CSV')
                    if r['stage'] not in stages or r['sample_id'] not in valid_ids:
                        raise ValueError('Event not in this dataset/protocol')
                    expected=identities['baseline' if r['stage']=='baseline' else 'attack']
                    if r['run_id']!=expected:raise ValueError('Different dataset/model/prompt/checkpoint; use a new output directory')
                    if r['status'] not in {'OK','ERROR'}:raise ValueError('Invalid event status')
                    if r['artifact']:self.artifact(r)
                    self.events[(r['sample_id'],r['stage'])]=r

    def artifact(self,event):
        path=(self.root/event['artifact']).resolve()
        if not path.is_relative_to(self.root.resolve()):raise ValueError('Artifact escapes experiment directory')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=event['artifact_sha256']:raise ValueError('Saved artifact hash mismatch')
        return json.loads(raw)

    def get(self,sid,stage):return self.events.get((sid,stage))

    def ok(self,sid,stage):
        e=self.get(sid,stage)
        return e if e and e['status']=='OK' else None

    def text(self,sid,stage):return self.artifact(self.ok(sid,stage))['text']

    def checkpoint(self,state,selected,stage=None):
        events=[e for (sid,_),e in self.events.items() if sid in selected]
        atomic_json(self.folder/'checkpoint.json',dict(state=state,identities=self.ids,
            selected_count=len(selected),selected_sample_ids=selected,stage=stage,
            ok_stages=sum(e['status']=='OK' for e in events),error_stages=sum(e['status']=='ERROR' for e in events),
            authoritative_log='events.csv',updated_utc=api.utc_now()))

    def commit(self,event,artifact,selected):
        if artifact is not None:
            path=self.folder/'responses'/(uuid.uuid4().hex+'.json')
            atomic_json(path,artifact)
            event['artifact']=str(path.relative_to(self.root))
            event['artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        exists=self.path.exists() and self.path.stat().st_size>0
        self.folder.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            if not exists:w.writeheader()
            w.writerow(event);f.flush();os.fsync(f.fileno())
        self.events[(event['sample_id'],event['stage'])]=event
        self.checkpoint('running',selected,event['stage'])


def execute_stage(ledger,rows,stage,defender,attacker,prompts,bank,args):
    selected=[r['sample_id'] for r in args.selected]
    pending=[r for r in rows if not ledger.ok(r['sample_id'],stage) and
             (not ledger.get(r['sample_id'],stage) or args.retry_errors)]
    measured=0
    print(f'{ledger.folder.name}: {stage}, pending={len(pending)}',flush=True)
    for n,row in enumerate(pending,1):
        start=time.perf_counter();sid=row['sample_id'];artifact=None
        event=dict.fromkeys(FIELDS,'')
        event.update(run_id=ledger.ids['baseline' if stage=='baseline' else 'attack'],sample_id=sid,stage=stage)
        try:
            if stage in {'rewrite1','rewrite2'}:
                cached=bank.get(sid) if args.reuse_step1 and stage=='rewrite1' else None
                if cached:
                    if cached['original_text_sha256']!=hashlib.sha256(row['email_text'].encode()).hexdigest():
                        raise ValueError('Historical bank original text hash mismatch')
                    text=validate_rewrite(cached['rewrite_text'])
                    artifact={'text':text,'historical_provenance':cached}
                    event.update(source='historical_unverified',request_sha256=fingerprint(cached))
                else:
                    first=ledger.text(sid,'rewrite1') if stage=='rewrite2' else None
                    prompt=(prompts['ATTACKER_PROMPT_2_TEMPLATE'].format(original_email=row['email_text'],first_modified=first)
                            if stage=='rewrite2' else prompts['ATTACKER_PROMPT_1'])
                    payload=generation_payload(attacker,prompt,first if first is not None else row['email_text'])
                    request_hash=fingerprint({'contract':attacker,'payload':payload})
                    cache=ledger.root/'generation_cache'/(request_hash+'.json')
                    with file_lock(cache.with_suffix('.lock')):
                        if cache.exists():
                            artifact=json.loads(cache.read_text(encoding='utf-8'))
                            if artifact['request_sha256']!=request_hash or artifact['request']!=payload:
                                raise ValueError('Generation cache request mismatch')
                            if parse_generation(attacker,artifact['response'])!=artifact['text']:
                                raise ValueError('Generation cache response mismatch')
                            event['source']='shared_cache'
                        else:
                            response,_,elapsed,_=api.request_json(attacker['url'],credentials(attacker),payload,args.timeout,args.attempts)
                            artifact={'request':payload,'response':response,'request_sha256':request_hash}
                            text=parse_generation(attacker,response)
                            artifact['text']=text
                            atomic_json(cache,artifact)
                            event.update(source='generated',latency_ms=elapsed)
                    event['request_sha256']=request_hash
            else:
                text=row['email_text'] if stage=='baseline' else ledger.text(sid,'rewrite1' if stage=='classify1' else 'rewrite2')
                payload=defender_payload(defender,text)
                event['request_sha256']=fingerprint(payload)
                response,_,elapsed,_=api.request_json(defender['url'],credentials(defender),payload,args.timeout,args.attempts)
                artifact={'request':payload,'response':response}
                event['latency_ms']=elapsed
                label,p_spam,p_not_spam=parse_decision(defender,response,text)
                event.update(label=label,p_spam=p_spam,p_not_spam=p_not_spam,latency_ms=elapsed,source='inference')
            event['status']='OK'
        except Exception as exc:
            event.update(status='ERROR',error=str(exc)[:1000])
        ledger.commit(event,artifact,selected)
        if stage!='baseline':
            save_individual_rewrite(ledger,row,1 if stage.endswith('1') else 2)
        measured+=time.perf_counter()-start
        print(f'{sid} {stage} {event["status"]} {event["label"]} | '+api.progress(n,len(pending),measured,args.delay),flush=True)
        if args.delay and n<len(pending):time.sleep(args.delay)
    if stage!='baseline':export_rewrites(ledger,args.selected)


def report(ledger,rows,baseline_only):
    counts=dict(TP=0,TN=0,FP=0,FN=0,baseline_errors=0,eligible=0,step1_flips=0,step2_additional_flips=0,
                attack_unresolved=0,still_spam=0)
    results=[]
    classification_times=[]
    for r in rows:
        sid=r['sample_id'];b=ledger.ok(sid,'baseline');c1=ledger.ok(sid,'classify1');c2=ledger.ok(sid,'classify2')
        state='baseline_error';eligible=False
        if not b:counts['baseline_errors']+=1
        else:
            truth=r['true_label']=='SPAM';pred=b['label']=='SPAM'
            counts['TP' if truth and pred else 'TN' if not truth and not pred else 'FP' if pred else 'FN']+=1
            eligible=truth and pred
            state='not_eligible'
            if eligible:
                counts['eligible']+=1
                if c1 and c1['label']=='NOT_SPAM':counts['step1_flips']+=1;state='flip_step1'
                elif c1 and c2:
                    if c2['label']=='NOT_SPAM':counts['step2_additional_flips']+=1;state='flip_step2'
                    else:counts['still_spam']+=1;state='still_spam'
                else:counts['attack_unresolved']+=1;state='attack_pending_or_error'
        # Derive timing from saved successful classification events so resume
        # preserves measured values and never counts attacker/cache time.
        timings={}
        for column,event in [('baseline_time_sec',b),('step1_time_sec',c1),('step2_time_sec',c2)]:
            elapsed=event.get('latency_ms','') if event else ''
            timings[column]=round(float(elapsed)/1000,6) if elapsed not in ('',None) else ''
        measured=[value for value in timings.values() if value!='']
        classification_times.extend(measured)
        timings['classification_time_sec']=round(sum(measured),6) if measured else ''
        results.append(dict(sample_id=sid,true_label=r['true_label'],baseline_label=b['label'] if b else '',
                            eligible=int(eligible),step1_label=c1['label'] if c1 else '',step2_label=c2['label'] if c2 else '',
                            state=state,**timings))
    valid=sum(counts[k] for k in ['TP','TN','FP','FN']);tp,tn,fp,fn=[counts[k] for k in ['TP','TN','FP','FN']]
    div=lambda a,b:a/b if b else None
    flips=counts['step1_flips']+counts['step2_additional_flips'];eligible=counts['eligible']
    f1s=div(2*tp,2*tp+fp+fn);f1h=div(2*tn,2*tn+fp+fn)
    complete=not counts['baseline_errors'] and (baseline_only or not counts['attack_unresolved'])
    metrics=dict(**counts,selected_rows=len(rows),baseline_valid=valid,clean_accuracy=div(tp+tn,valid),
                 spam_recall=div(tp,tp+fn),false_positive_rate=div(fp,fp+tn),
                 macro_f1=(f1s+f1h)/2 if f1s is not None and f1h is not None else None,
                 observed_flips=flips,flip_rate_lower_bound=div(flips,eligible),
                 flip_rate_upper_bound=div(flips+counts['attack_unresolved'],eligible),
                 cumulative_label_flip_rate=div(flips,eligible) if not counts['attack_unresolved'] else None,
                 semantic_validity='NOT_ASSESSED',evaluation='baseline_only' if baseline_only else 'two_step',
                 classification_timed_requests=len(classification_times),
                 classification_total_time_sec=round(sum(classification_times),6) if classification_times else None,
                 classification_mean_time_sec=round(sum(classification_times)/len(classification_times),6) if classification_times else None,
                 complete_for_selection=complete,identities=ledger.ids)
    metrics.update(paper_evaluation.attack_metrics(ledger,rows,counts,baseline_only))
    export_rewrites(ledger,rows)
    atomic_json(ledger.folder/'metrics.json',metrics)
    for name,data in [('sample_results.csv',results),('metrics.csv',[{k:json.dumps(v) if isinstance(v,dict) else v for k,v in metrics.items()}])]:
        target=ledger.folder/name;tmp=target.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        os.replace(tmp,target)
    return complete


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'decision_models/adaptive_main.json')
    p.add_argument('--model',default='all',help='Any defender key in the configuration, or all')
    p.add_argument('--dataset',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--pilot',action='store_true',help='First 5 spam and 5 ham; adaptive stages only for eligible spam')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--reuse-step1',action='store_true',help='Declared hybrid mode: use unverified historical Step-1 generation where available')
    p.add_argument('--baseline-from',type=Path,help='Import matching saved baseline events from this run root')
    p.add_argument('--resume',action='store_true',help='Automatic even without this flag')
    p.add_argument('--retry-errors',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--check-server',action='store_true')
    p.add_argument('--timeout',type=float,default=300);p.add_argument('--attempts',type=int,default=3)
    p.add_argument('--delay',type=float,default=0)
    return p


def main(argv=None,config_override=None):
    args=parser().parse_args(argv)
    if args.limit<0 or args.attempts<1 or not math.isfinite(args.timeout) or args.timeout<=0 or not math.isfinite(args.delay) or args.delay<0:
        raise ValueError('Invalid limit/retry/timeout/delay')
    if args.pilot and args.limit:raise ValueError('Choose pilot or limit, not both')
    config=copy.deepcopy(config_override) if config_override is not None else json.loads(args.config.read_text(encoding='utf-8-sig'))
    dataset=rooted(args.dataset or config['dataset']);output=rooted(args.output_dir or config['output_dir'])
    source=rooted(args.baseline_from) if args.baseline_from else None
    if source==output:raise ValueError('Baseline source and output must differ')
    if config['profile']=='experimenting' and not output.is_relative_to((ROOT/'experimenting').resolve()):
        raise ValueError('Experimenting output must remain inside experimenting')
    rows=api.load_dataset(dataset,{'clean'})
    if not rows:raise ValueError('No baseline inputs')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Baseline sample_id must be unique per original row')
    args.selected=([r for r in rows if r['true_label']=='SPAM'][:5]+[r for r in rows if r['true_label']=='NOT_SPAM'][:5]
                   if args.pilot else rows[:args.limit] if args.limit else rows)
    bank_path=rooted(config['step1_bank'])
    bank_rows=read_csv(bank_path) if args.reuse_step1 else []
    bank={r['sample_id']:r for r in bank_rows}
    if len(bank)!=len(bank_rows):raise ValueError('Historical bank must have one candidate per sample')
    prompts=json.loads((ROOT/'decision_models/paper_prompts.json').read_text(encoding='utf-8'))
    attacker=config['attacker']
    for key,env in [('url','ATTACKER_URL'),('model','ATTACKER_MODEL'),('revision','ATTACKER_REVISION'),('key_env','ATTACKER_API_KEY_ENV')]:
        if os.environ.get(env):attacker[key]=os.environ[env]
    for key in ['temperature','max_tokens']:
        if not isinstance(attacker.get(key),(int,float)) or not math.isfinite(attacker[key]):raise ValueError(f'Invalid attacker {key}')
    if attacker['max_tokens']<1 or attacker['temperature']<0:raise ValueError('Invalid generation settings')
    names=selected_models(config,args.model)
    plans=[]
    # Resolve every selected model/revision before making any inference request.
    for name in names:
        validate_model_key(name)
        if name not in config['defenders']:raise ValueError(f'Model {name} is not configured in this profile')
        defender=config['defenders'][name]
        for key,suffix in [('revision','CHECKPOINT_ID'),('model','MODEL_ID'),('url','URL'),('tool_id','TOOL_ID')]:
            if os.environ.get(name.upper()+'_'+suffix):defender[key]=os.environ[name.upper()+'_'+suffix]
        if name=='pplx' and defender.get('transport')=='direct' and defender.get('template') is None:
            raise ValueError('Direct PPLX requires a verified native template containing __EMAIL_TEXT__')
        # Validate the complete payload before any model starts inference.
        defender_payload(defender,'__PREFLIGHT_EMAIL__')
        validate_endpoint(defender,live=not args.dry_run,timeout=args.timeout)
        base_id=fingerprint({'dataset':hashlib.sha256(dataset.read_bytes()).hexdigest(),'defender':defender,
                             'questions':api.SYSTEMONE_QUESTION,'version':1})
        plans.append((name,defender,base_id))
    validate_endpoint(attacker,attacker=True,live=not args.dry_run and not args.baseline_only,timeout=args.timeout)
    attack_contract=dict(attacker=attacker,prompts=prompts,reuse_step1=args.reuse_step1,
                         bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest() if args.reuse_step1 else None,version=1)
    ledgers=[]
    for name,defender,base_id in plans:
        ids={'baseline':base_id,'attack':fingerprint({'baseline':base_id,**attack_contract})}
        ledger=Ledger(output,name,ids,{r['sample_id'] for r in rows})
        if source:
            from gan_pipeline import baseline_source
            baseline_source(source,name,ids['baseline'],{r['sample_id'] for r in rows})
        ledgers.append((ledger,defender))
        print(f'{name}: {defender["url"]} | model={defender["model"] or "UNCONFIGURED"} | baseline={len(rows)} selected={len(args.selected)} | output={ledger.folder}')
    if args.dry_run:
        print('DRY RUN: no network/inference/output changes. Live runs require model IDs, revisions and credentials.')
        return 0
    if args.check_server:
        print('Endpoint/model metadata and resume checks passed; no classification/generation performed.')
        return 0
    for ledger,defender in ledgers:
        with file_lock(ledger.folder/'run.lock'):
            # Reload under the lock to avoid racing another invocation's completed work.
            ledger=Ledger(output,ledger.folder.name,ledger.ids,{r['sample_id'] for r in rows})
            selected_ids=[r['sample_id'] for r in args.selected]
            atomic_json(ledger.folder/'run.json',dict(defender=defender,attack_contract=attack_contract,
                dataset=str(dataset),dataset_rows=len(rows),selected_sample_ids=selected_ids,
                baseline_only=args.baseline_only,identities=ledger.ids,
                experiment_purpose=config.get('experiment_purpose','paper_benchmark')))
            ledger.checkpoint('running',selected_ids)
            try:
                if source and (source/ledger.folder.name/'run.json').exists():
                    from gan_pipeline import baseline_source,import_baseline
                    with file_lock(source/ledger.folder.name/'run.lock'):
                        prior=baseline_source(source,ledger.folder.name,ledger.ids['baseline'],{r['sample_id'] for r in rows})
                        if prior:import_baseline(prior,ledger,args.selected,args.retry_errors)
                execute_stage(ledger,args.selected,'baseline',defender,attacker,prompts,bank,args)
                if not args.baseline_only:
                    eligible=[r for r in args.selected if r['true_label']=='SPAM' and ledger.ok(r['sample_id'],'baseline')
                              and ledger.ok(r['sample_id'],'baseline')['label']=='SPAM']
                    execute_stage(ledger,eligible,'rewrite1',defender,attacker,prompts,bank,args)
                    execute_stage(ledger,[r for r in eligible if ledger.ok(r['sample_id'],'rewrite1')],
                                  'classify1',defender,attacker,prompts,bank,args)
                    second=[r for r in eligible if ledger.ok(r['sample_id'],'classify1') and
                            ledger.ok(r['sample_id'],'classify1')['label']=='SPAM']
                    execute_stage(ledger,second,'rewrite2',defender,attacker,prompts,bank,args)
                    execute_stage(ledger,[r for r in second if ledger.ok(r['sample_id'],'rewrite2')],
                                  'classify2',defender,attacker,prompts,bank,args)
            except KeyboardInterrupt:
                report(ledger,args.selected,args.baseline_only)
                ledger.checkpoint('interrupted',selected_ids)
                print('Interrupted. Completed stages are saved; rerun with --resume.')
                return 130
            complete=report(ledger,args.selected,args.baseline_only)
            ledger.checkpoint('complete' if complete else 'incomplete',selected_ids)
            if not complete:
                print('Incomplete stages remain. Inspect events.csv; use --retry-errors for saved failures.')
                return 2
    return 0


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        from env_config import load_project_env
        load_project_env()
        sys.exit(main())
    except (ValueError,OSError,KeyError) as exc:
        print(f'Configuration/data error: {exc}',file=sys.stderr);sys.exit(2)
