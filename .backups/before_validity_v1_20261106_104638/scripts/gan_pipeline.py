"""Resumable OpenAttack SST GAN experiment using main decision-model adapters."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import adaptive_pipeline as core
import paper_evaluation as metrics
from gan_engine import GanEngine, gan_contract

STAGES=('baseline','gan_probe','gan_attack','gan_classify')
CSV_FIELDS=['defender','sample_id','true_label','original_text','baseline_label','candidate_text',
    'candidate_label','candidate_status','returned_text','used_text','is_modified','attack_status',
    'return_status','predicted_label','classification_status','label_flip','baseline_time_sec',
    'probe_time_sec','classification_time_sec','gan_search_elapsed_sec','error']


class ClassificationFailure(ValueError):
    pass


class GanLedger(core.Ledger):
    def __init__(self,root,model,identities,valid_ids):
        super().__init__(root,model,identities,valid_ids,stages=STAGES)


def event_for(ledger,sid,stage):
    event=dict.fromkeys(core.FIELDS,'')
    event.update(run_id=ledger.ids['baseline' if stage=='baseline' else 'attack'],sample_id=sid,stage=stage)
    return event


def classify(ledger,row,stage,text,defender,args):
    sid=row['sample_id'];prior=ledger.get(sid,stage)
    payload=core.defender_payload(defender,text)
    if prior and prior['status']=='OK':
        if ledger.artifact(prior)['request']!=payload:raise ValueError('GAN replay candidate/request changed; use a new output directory')
        return prior['label']
    if prior and not args.retry_errors:raise ClassificationFailure('Saved classification error; use --retry-errors')
    event=event_for(ledger,sid,stage);event.update(source='inference',request_sha256=core.fingerprint(payload))
    artifact={'request':payload,'text':text}
    try:
        response,status,elapsed,_=core.api.request_json(defender['url'],core.credentials(defender),payload,args.timeout,args.attempts)
        artifact['response']=response
        event['latency_ms']=elapsed
        label,p_spam,p_not_spam=core.parse_decision(defender,response,text)
        event.update(status='OK',label=label,p_spam=p_spam,p_not_spam=p_not_spam)
    except Exception as exc:event.update(status='ERROR',error=str(exc)[:1000])
    ledger.commit(event,artifact,[r['sample_id'] for r in args.selected])
    save_individual(ledger,row)
    if event['status']!='OK':raise ClassificationFailure(f'{stage} failed: {event["error"]}')
    return event['label']


def run_attack(ledger,row,defender,engine,args):
    sid=row['sample_id'];prior=ledger.get(sid,'gan_attack')
    if prior and (prior['status']=='OK' or not args.retry_errors):return
    event=event_for(ledger,sid,'gan_attack');start=time.perf_counter()
    artifact={'original_sha256':hashlib.sha256(row['email_text'].encode()).hexdigest()}
    class Victim:
        calls=0
        def get_pred(self,texts):
            # Source default SST path makes one candidate query. Reject an
            # unexpected algorithm change rather than silently losing queries.
            if len(texts)!=1 or self.calls:raise ValueError('Unexpected query count for pinned SST GAN')
            self.calls+=1
            import numpy as np
            label=classify(ledger,row,'gan_probe',texts[0],defender,args)
            return np.array([1 if label=='SPAM' else 0],dtype=np.int64)
    try:
        returned=engine.attack(row['email_text'],sid,Victim())
        if returned is not None and not isinstance(returned,str):raise ValueError('Unexpected GAN return type')
        returned=returned if returned and returned.strip() else ''
        probe=ledger.ok(sid,'gan_probe')
        if not probe:raise ValueError('GAN completed without a classified candidate')
        candidate=ledger.artifact(probe)['text']
        if returned and returned!=candidate:raise ValueError('GAN returned text differs from recorded candidate')
        artifact.update(candidate_text=candidate,returned_text=returned,
                        used_text=returned or row['email_text'],is_modified=bool(returned and returned.strip()!=row['email_text'].strip()),
                        return_status='RETURNED' if returned else 'NO_ADVERSARIAL_FOUND')
        event.update(status='OK',source='openattack_gan_sst',request_sha256=core.fingerprint(
            dict(original=row['email_text'],sample_id=sid,contract=engine.contract)))
    except Exception as exc:event.update(status='ERROR',error=str(exc)[:1000])
    event['latency_ms']=round((time.perf_counter()-start)*1000,2)
    ledger.commit(event,artifact,[r['sample_id'] for r in args.selected])
    save_individual(ledger,row)


def record(ledger,row):
    sid=row['sample_id'];base=ledger.get(sid,'baseline') or {};probe=ledger.get(sid,'gan_probe') or {}
    attack=ledger.get(sid,'gan_attack') or {};final=ledger.get(sid,'gan_classify') or {}
    result=ledger.artifact(attack) if attack.get('artifact') else {}
    proposed=ledger.artifact(probe).get('text','') if probe.get('artifact') else ''
    seconds=lambda e:round(float(e['latency_ms'])/1000,6) if e.get('latency_ms') not in ('',None) else ''
    flipped=bool(result.get('is_modified') and final.get('status')=='OK' and final['label']=='NOT_SPAM')
    return dict(defender=ledger.folder.name,sample_id=sid,true_label=row['true_label'],original_text=row['email_text'],
        baseline_label=base.get('label',''),candidate_text=proposed,candidate_label=probe.get('label',''),
        candidate_status=probe.get('status','PENDING'),returned_text=result.get('returned_text',''),used_text=result.get('used_text',''),
        is_modified=int(bool(result.get('is_modified'))),attack_status=attack.get('status','PENDING'),return_status=result.get('return_status',''),
        predicted_label=final.get('label',''),classification_status=final.get('status','PENDING'),label_flip=int(flipped),
        baseline_time_sec=seconds(base),probe_time_sec=seconds(probe),classification_time_sec=seconds(final),
        gan_search_elapsed_sec=seconds(attack),error=' | '.join(e['error'] for e in (probe,attack,final) if e.get('error')))


def save_individual(ledger,row):
    if not any(ledger.get(row['sample_id'],s) for s in STAGES[1:]):return
    import re
    name=re.sub('[^a-zA-Z0-9_-]','_',row['sample_id'])[:60]+'_'+hashlib.sha256(row['sample_id'].encode()).hexdigest()[:16]+'.csv'
    core.atomic_csv(ledger.folder/'rewrites/gan'/name,[record(ledger,row)],CSV_FIELDS)


def report(ledger,rows,baseline_only=False):
    counts=dict(TP=0,TN=0,FP=0,FN=0,baseline_errors=0)
    rewrites=[];samples=[];times=[]
    for row in rows:
        sid=row['sample_id'];base=ledger.ok(sid,'baseline')
        if base:
            truth=row['true_label']=='SPAM';pred=base['label']=='SPAM'
            counts['TP' if truth and pred else 'TN' if not truth and not pred else 'FP' if pred else 'FN']+=1
            eligible=truth and pred
        else:counts['baseline_errors']+=1;eligible=False
        for stage in ('baseline','gan_probe','gan_classify'):
            e=ledger.ok(sid,stage)
            if e and e['latency_ms'] not in ('',None):times.append(float(e['latency_ms'])/1000)
        result=record(ledger,row)
        samples.append(result)
        if eligible:
            rewrites.append(result);save_individual(ledger,row)
    n=len(rewrites);flips=sum(r['label_flip'] for r in rewrites)
    done=sum(r['attack_status']=='OK' and r['classification_status']=='OK' for r in rewrites)
    modified=sum(r['is_modified'] for r in rewrites)
    complete=not counts['baseline_errors'] and (baseline_only or done==n)
    rate=metrics.percent(flips,n) if complete and not baseline_only else None
    tp,tn,fp,fn=(counts[k] for k in ('TP','TN','FP','FN'))
    s_f1=metrics.divide(2*tp,2*tp+fp+fn);h_f1=metrics.divide(2*tn,2*tn+fp+fn)
    summary=dict(**counts,**metrics.classification_metrics(counts),attack_kind='gan',selected_rows=len(rows),
        baseline_valid=tp+tn+fp+fn,clean_accuracy=metrics.divide(tp+tn,tp+tn+fp+fn),spam_recall=metrics.divide(tp,tp+fn),
        false_positive_rate=metrics.divide(fp,fp+tn),macro_f1=(s_f1+h_f1)/2 if s_f1 is not None and h_f1 is not None else None,
        eligible=n,gan_completed=done,gan_unresolved=n-done,gan_modified=modified,
        gan_no_adversarial_found=sum(r['return_status']=='NO_ADVERSARIAL_FOUND' for r in rewrites),
        gan_successes=flips,gan_asr_pct=rate,gan_robust_accuracy_pct=100-rate if rate is not None else None,
        gan_asr_lower_bound_pct=metrics.percent(flips,n),gan_asr_upper_bound_pct=metrics.percent(flips+n-done,n),
        gan_modified_asr_pct=metrics.percent(flips,modified) if complete and not baseline_only else None,
        fallback_label_drift=sum(r['return_status']=='NO_ADVERSARIAL_FOUND' and r['classification_status']=='OK'
                                and r['predicted_label']=='NOT_SPAM' for r in rewrites),
        classification_timed_requests=len(times),classification_total_time_sec=sum(times) if times else None,
        classification_mean_time_sec=sum(times)/len(times) if times else None,
        baseline_reused=sum(bool(ledger.ok(r['sample_id'],'baseline')) and ledger.ok(r['sample_id'],'baseline')['source']=='baseline_reuse' for r in rows),
        complete_for_selection=complete,evaluation='baseline_only' if baseline_only else 'gan_sst',
        semantic_validity='NOT_ASSESSED',identities=ledger.ids)
    core.atomic_json(ledger.folder/'metrics.json',summary)
    flat={k:json.dumps(v) if isinstance(v,dict) else v for k,v in summary.items()}
    core.atomic_csv(ledger.folder/'metrics.csv',[flat],list(flat))
    core.atomic_csv(ledger.folder/'sample_results.csv',samples,CSV_FIELDS)
    core.atomic_csv(ledger.folder/'gan_rewrites.csv',rewrites,CSV_FIELDS)
    return complete


def baseline_source(root,name,baseline_id,valid_ids):
    path=root/name/'run.json'
    if not path.exists():return None
    saved=json.loads(path.read_text(encoding='utf-8'))
    if saved['identities']['baseline']!=baseline_id:
        raise ValueError(f'{name}: baseline source uses a different dataset/model/endpoint/revision; choose the matching source or --no-reuse-baseline')
    cls=GanLedger if saved.get('attack_kind')=='gan' else core.Ledger
    return cls(root,name,saved['identities'],valid_ids)


def import_baseline(source,ledger,rows,retry_errors=False):
    for row in rows:
        sid=row['sample_id'];prior=ledger.get(sid,'baseline');old=source.ok(sid,'baseline')
        if not old or (prior and (prior['status']=='OK' or not retry_errors)):continue
        artifact=copy.deepcopy(source.artifact(old))
        artifact['baseline_reused_from']=dict(root=str(source.root),artifact=old['artifact'],sha256=old['artifact_sha256'])
        event={**old,'source':'baseline_reuse','artifact':'','artifact_sha256':''}
        ledger.commit(event,artifact,[r['sample_id'] for r in rows])


def needs_attack_work(ledger,sid,retry_errors=False):
    attack=ledger.get(sid,'gan_attack')
    if not attack:return True
    if attack['status']!='OK':return retry_errors
    final=ledger.get(sid,'gan_classify')
    return not final or (final['status']!='OK' and retry_errors)


def main(argv=None,config_override=None):
    parser=core.parser();parser.description=__doc__
    parser.add_argument('--no-reuse-baseline',action='store_true')
    parser.add_argument('--gan-resource-dir',type=Path)
    parser.add_argument('--gan-seed',type=int)
    args=parser.parse_args(argv)
    if args.reuse_step1:raise ValueError('--reuse-step1 applies to LLM attacks, not GAN')
    if args.limit<0 or args.attempts<1 or not math.isfinite(args.timeout) or args.timeout<=0 or not math.isfinite(args.delay) or args.delay<0:
        raise ValueError('Invalid limit/retries/timeout/delay')
    if args.pilot and args.limit:raise ValueError('Choose pilot or limit')
    config=copy.deepcopy(config_override) if config_override is not None else json.loads(args.config.read_text(encoding='utf-8-sig'))
    dataset=core.rooted(args.dataset or config['dataset'])
    output=core.rooted(args.output_dir or config.get('gan_output_dir','outputs/paper3000_gan'))
    if config['profile']=='experimenting' and not output.is_relative_to((core.ROOT/'experimenting').resolve()):
        raise ValueError('Output must remain inside the selected profile folder')
    source=None if args.no_reuse_baseline else core.rooted(args.baseline_from or config['output_dir'])
    if source==output:raise ValueError('GAN and baseline source must use separate output directories')
    rows=core.api.load_dataset(dataset,{'clean'});valid_ids={r['sample_id'] for r in rows}
    if not rows or len(valid_ids)!=len(rows):raise ValueError('Dataset must contain unique original sample IDs')
    args.selected=([r for r in rows if r['true_label']=='SPAM'][:5]+[r for r in rows if r['true_label']=='NOT_SPAM'][:5]
                   if args.pilot else rows[:args.limit] if args.limit else rows)
    gan=copy.deepcopy(config.get('gan',{}))
    if args.gan_resource_dir:gan['resource_dir']=str(args.gan_resource_dir)
    if args.gan_seed is not None:gan['seed']=args.gan_seed
    if int(gan.get('seed',20261104))<0:raise ValueError('GAN seed must be non-negative')
    contract=gan_contract(gan,required=not args.dry_run and not args.baseline_only)
    plans=[]
    for name in core.selected_models(config,args.model):
        core.validate_model_key(name)
        if name not in config['defenders']:raise ValueError('Unknown defender: '+name)
        defender=config['defenders'][name]
        for key,suffix in [('revision','CHECKPOINT_ID'),('model','MODEL_ID'),('url','URL'),('tool_id','TOOL_ID')]:
            if os.environ.get(name.upper()+'_'+suffix):defender[key]=os.environ[name.upper()+'_'+suffix]
        if name=='pplx' and defender.get('transport')=='direct' and defender.get('template') is None:raise ValueError('Native PPLX requires a verified template')
        core.defender_payload(defender,'__PREFLIGHT_EMAIL__')
        core.validate_endpoint(defender,live=not args.dry_run,timeout=args.timeout)
        baseline=core.fingerprint(dict(dataset=hashlib.sha256(dataset.read_bytes()).hexdigest(),defender=defender,questions=core.api.SYSTEMONE_QUESTION,version=1))
        identities=dict(baseline=baseline,attack=core.fingerprint({'baseline':baseline,**contract}))
        ledger=GanLedger(output,name,identities,valid_ids)
        if source:baseline_source(source,name,baseline,valid_ids)
        plans.append((name,defender,identities))
        print(f'{name}: GAN SST | {defender["url"]} | baseline={len(rows)} selected={len(args.selected)} | output={ledger.folder}')
    if args.dry_run:print('DRY RUN: no model calls, GAN load, downloads, or output changes.');return 0
    engine=None if args.baseline_only else GanEngine(gan)
    if args.check_server:print('GAN resources, model configuration, and resume logs checked; no inference.');return 0
    for name,defender,identities in plans:
        with core.file_lock(output/name/'run.lock'):
            ledger=GanLedger(output,name,identities,valid_ids);selected_ids=[r['sample_id'] for r in args.selected]
            core.atomic_json(ledger.folder/'run.json',dict(attack_kind='gan',defender=defender,attack_contract=contract,
                dataset=str(dataset),dataset_rows=len(rows),selected_sample_ids=selected_ids,baseline_only=args.baseline_only,identities=identities,
                experiment_purpose=config.get('experiment_purpose','paper_benchmark')))
            try:
                if source and (source/name/'run.json').exists():
                    with core.file_lock(source/name/'run.lock'):
                        prior=baseline_source(source,name,identities['baseline'],valid_ids)
                        if prior:import_baseline(prior,ledger,args.selected,args.retry_errors)
                core.execute_stage(ledger,args.selected,'baseline',defender,{}, {},{},args)
                if not args.baseline_only:
                    eligible=[r for r in args.selected if r['true_label']=='SPAM' and ledger.ok(r['sample_id'],'baseline')
                              and ledger.ok(r['sample_id'],'baseline')['label']=='SPAM']
                    pending=[r for r in eligible if needs_attack_work(ledger,r['sample_id'],args.retry_errors)]
                    print(f'{name}: {len(pending)} pending GAN samples of {len(eligible)} eligible.',flush=True)
                    start=time.perf_counter()
                    for n,row in enumerate(pending,1):
                        run_attack(ledger,row,defender,engine,args)
                        event=ledger.ok(row['sample_id'],'gan_attack')
                        if event:
                            try:classify(ledger,row,'gan_classify',ledger.artifact(event)['used_text'],defender,args)
                            except ClassificationFailure:pass # The error remains explicit in the ledger.
                        print(f'{name} GAN {row["sample_id"]} | '+core.api.progress(n,len(pending),time.perf_counter()-start,0),flush=True)
                        if args.delay and n<len(pending):time.sleep(args.delay)
            except KeyboardInterrupt:
                report(ledger,args.selected,args.baseline_only);ledger.checkpoint('interrupted',selected_ids)
                print('GAN interrupted. Committed stages and candidate probes are saved.');return 130
            complete=report(ledger,args.selected,args.baseline_only)
            ledger.checkpoint('complete' if complete else 'incomplete',selected_ids)
            if not complete:print('Incomplete GAN results; inspect events.csv and resume with --retry-errors.');return 2
    return 0


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        from env_config import load_project_env
        load_project_env()
        sys.exit(main())
    except (ValueError,OSError,KeyError,ImportError) as exc:
        print('GAN configuration/data error:',exc,file=sys.stderr);sys.exit(2)
