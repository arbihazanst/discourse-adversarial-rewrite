"""Two-step validity-gated experiment, shared by LLM and GAN branches."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import adaptive_pipeline as core
import rewrite_validity as validity

STAGES=('baseline','rewrite1','validate1','classify1','rewrite2','validate2','classify2')
TERMINAL={'INVALID','UNCERTAIN','NO_CHANGE'}


class ValidityLedger(core.Ledger):
    validity_gated=True
    def __init__(self,root,model,identities,valid_ids):
        super().__init__(root,model,identities,valid_ids,stages=STAGES)


def configuration(config):
    evaluator=config.get('rewrite_evaluator',{})
    if evaluator.get('transport')!='tool' or evaluator.get('role')!='pplx_validity':
        raise ValueError('Configure rewrite_evaluator as the PPLX tool with role=pplx_validity')
    if config['profile']=='main' and (set(config['defenders'])-{'clef','d3'}):
        raise ValueError('Main defenders must be Clef/d3 only; PPLX belongs in rewrite_evaluator')
    if config['profile']=='main' and any(d.get('transport') not in {'direct','openwebui'} for d in config['defenders'].values()):
        raise ValueError('Main Clef/d3 use direct System One or Open WebUI Pipe, not the PPLX tool')
    if config['profile']=='experimenting' and set(config['defenders'])!={'pplx'}:
        raise ValueError('This experimenting protocol uses PPLX only as defender; Clef/d3 run on main')


def seconds(event):
    return round(float(event['latency_ms'])/1000,6) if event and event.get('latency_ms') not in ('',None) else ''


def record(ledger,row,step):
    sid=row['sample_id']; suffix=str(step)
    generated=ledger.get(sid,'rewrite'+suffix) or {}
    evaluated=ledger.get(sid,'validate'+suffix) or {}
    classified=ledger.get(sid,'classify'+suffix) or {}
    artifact=ledger.artifact(evaluated) if evaluated.get('artifact') else {}
    verdict=evaluated.get('label') or evaluated.get('status','PENDING')
    original=core.rewrite_record(ledger,row,step)
    original.update(validation_status=verdict,validation_reason=artifact.get('reason',''),
        validation_time_sec=seconds(evaluated),validation_error=evaluated.get('error',''),
        rubric_version=validity.VERSION,
        classification_status=classified.get('status','SKIPPED_'+verdict if verdict in TERMINAL else 'PENDING'))
    for key in validity.RUBRIC: original['validity_'+key]=artifact.get('checks',{}).get(key,{}).get('label','')
    original['step_elapsed_sec']=round(sum(v for v in (seconds(generated),seconds(evaluated),seconds(classified)) if v!=''),6)
    return original


def save_individual(ledger,row,step):
    result=record(ledger,row,step)
    safe=hashlib.sha256(row['sample_id'].encode()).hexdigest()[:16]
    core.atomic_csv(ledger.folder/'rewrites'/('step'+str(step))/(safe+'.csv'),[result],list(result))


def export(ledger,rows):
    all_records=[]
    fields=list(record(ledger,rows[0],1))
    for step in (1,2):
        records=[]
        for row in rows:
            sid=row['sample_id']
            if not ledger.get(sid,'rewrite'+str(step)): continue
            result=record(ledger,row,step); records.append(result); all_records.append(result)
        core.atomic_csv(ledger.folder/('step'+str(step)+'_rewrites.csv'),records,fields)
    return all_records


def report(ledger,rows,baseline_only=False):
    records=export(ledger,rows)
    counts=dict(TP=0,TN=0,FP=0,FN=0,baseline_errors=0,eligible=0,step1_flips=0,step2_additional_flips=0,
                attack_unresolved=0,still_spam=0,invalid=0,uncertain=0,no_change=0)
    results=[]
    for row in rows:
        sid=row['sample_id'];base=ledger.ok(sid,'baseline'); eligible=False;state='baseline_error_or_pending'
        if not base: counts['baseline_errors']+=1
        else:
            truth=row['true_label']=='SPAM';prediction=base['label']=='SPAM'
            counts['TP' if truth and prediction else 'TN' if not truth and not prediction else 'FP' if prediction else 'FN']+=1
            eligible=truth and prediction;state='not_eligible'
            if eligible:
                counts['eligible']+=1;state='baseline_only' if baseline_only else 'attack_pending_or_error'
                if not baseline_only:
                    for step in (1,2):
                        val=ledger.ok(sid,'validate'+str(step));decision=ledger.ok(sid,'classify'+str(step))
                        if not val: break
                        if val['label'] in TERMINAL:
                            counts[val['label'].lower()]+=1;state=val['label'].lower()+'_step'+str(step);break
                        if val['label']!='VALID' or not decision: break
                        if decision['label']=='NOT_SPAM':
                            counts['step1_flips' if step==1 else 'step2_additional_flips']+=1;state='flip_step'+str(step);break
                        if step==2: counts['still_spam']+=1;state='still_spam'
                    if state=='attack_pending_or_error': counts['attack_unresolved']+=1
        result=dict(sample_id=sid,true_label=row['true_label'],baseline_label=base['label'] if base else '',eligible=int(eligible),state=state)
        for step in (1,2):
            val=ledger.get(sid,'validate'+str(step)) or {};decision=ledger.ok(sid,'classify'+str(step)) or {}
            result['step'+str(step)+'_validation']=val.get('label') or val.get('status','NOT_REACHED')
            result['step'+str(step)+'_label']=decision.get('label','')
        for stage in STAGES: result[stage+'_time_sec']=seconds(ledger.get(sid,stage))
        result['total_stage_elapsed_sec']=round(sum(v for k,v in result.items() if k.endswith('_time_sec') and v!=''),6)
        results.append(result)
    div=lambda a,b:a/b if b else None
    steps={}
    for step in (1,2):
        subset=[r for r in records if r['step']==step]
        stats={key:sum(r['validation_status']==key for r in subset) for key in ('VALID','INVALID','UNCERTAIN','NO_CHANGE','ERROR','PENDING')}
        stats.update(generated=sum(r['generation_status']=='OK' for r in subset),generation_errors=sum(r['generation_status']=='ERROR' for r in subset),
            classified=sum(r['classification_status']=='OK' for r in subset),classification_errors=sum(r['classification_status']=='ERROR' for r in subset),
            flips=sum(r['classification_status']=='OK' and r['predicted_label']=='NOT_SPAM' for r in subset))
        stats['acceptance_rate']=div(stats['VALID'],stats['VALID']+stats['INVALID'])
        stats['flip_rate_among_classified_valid']=div(stats['flips'],stats['classified'])
        stats['failed_criteria']={key:sum(r['validity_'+key]=='FAIL' for r in subset) for key in validity.RUBRIC}
        steps['step'+str(step)]=stats
    tp,tn,fp,fn=(counts[k] for k in ('TP','TN','FP','FN'));n=tp+tn+fp+fn
    flips=counts['step1_flips']+counts['step2_additional_flips'];eligible=counts['eligible']
    complete=counts['baseline_errors']==0 and (baseline_only or counts['attack_unresolved']==0)
    fs=div(2*tp,2*tp+fp+fn);fh=div(2*tn,2*tn+fp+fn)
    metrics=dict(**counts,selected_rows=len(rows),baseline_valid=n,clean_accuracy=div(tp+tn,n),
        spam_precision=div(tp,tp+fp),spam_recall=div(tp,tp+fn),false_positive_rate=div(fp,fp+tn),
        macro_f1=(fs+fh)/2 if fs is not None and fh is not None else None,
        observed_valid_flips=flips,overall_valid_attack_success_rate=div(flips,eligible) if complete and not baseline_only else None,
        flip_rate_lower_bound=div(flips,eligible) if not baseline_only else None,
        flip_rate_upper_bound=div(flips+counts['attack_unresolved'],eligible) if not baseline_only else None,
        semantic_validity='PPLX_RUBRIC_V1_NOT_HUMAN_VALIDATED',steps=steps,
        complete_for_selection=complete,identities=ledger.ids,protocol='validity_gated_v1')
    metrics.update(core.paper_evaluation.classification_metrics(counts))
    metrics['validity_gated_asr_pct']=100*metrics['overall_valid_attack_success_rate'] if metrics['overall_valid_attack_success_rate'] is not None else None
    metrics['validity_gated_robust_accuracy_pct']=100-metrics['validity_gated_asr_pct'] if metrics['validity_gated_asr_pct'] is not None else None
    core.atomic_csv(ledger.folder/'sample_results.csv',results,list(results[0]))
    core.atomic_json(ledger.folder/'metrics.json',metrics)
    core.atomic_csv(ledger.folder/'metrics.csv',[{k:json.dumps(v) if isinstance(v,dict) else v for k,v in metrics.items()}],list(metrics))
    core.atomic_csv(ledger.folder/'validity_summary.csv',[dict(step=k,**{n:json.dumps(v) if isinstance(v,dict) else v for n,v in s.items()}) for k,s in steps.items()],['step',*steps['step1']])
    return complete


def generate_gan(ledger,row,step,engine,args):
    sid=row['sample_id'];stage='rewrite'+str(step);prior=ledger.get(sid,stage)
    if prior and (prior['status']=='OK' or not args.retry_errors): return
    text=row['email_text'] if step==1 else ledger.text(sid,'rewrite1')
    event=dict.fromkeys(core.FIELDS,'');event.update(run_id=ledger.ids['attack'],sample_id=sid,stage=stage,source='gan_sst_candidate')
    start=time.perf_counter();artifact=None
    try:
        generated=engine.generate(text,sid+':step'+str(step))
        if not isinstance(generated,str): raise ValueError('GAN did not produce a text candidate')
        artifact=dict(text=generated,input_text=text,generation_policy='capture SST candidate before defender query',step=step)
        event.update(status='OK',request_sha256=core.fingerprint(dict(contract=engine.contract,input=text,sample_id=sid,step=step)))
    except Exception as exc: event.update(status='ERROR',error=str(exc)[:1000])
    event['latency_ms']=round((time.perf_counter()-start)*1000,3)
    ledger.commit(event,artifact,[r['sample_id'] for r in args.selected])


def main(argv=None,config_override=None,attack_kind='llm'):
    parser=core.parser()
    if attack_kind=='gan':
        parser.add_argument('--no-reuse-baseline',action='store_true');parser.add_argument('--gan-resource-dir',type=Path);parser.add_argument('--gan-seed',type=int)
    args=parser.parse_args(argv)
    if args.reuse_step1: raise ValueError('Validity-gated v1 uses fresh candidates; historical rewrite reuse is disabled')
    if args.pilot and args.limit: raise ValueError('Choose --pilot or --limit')
    if args.limit<0 or args.attempts<1 or not math.isfinite(args.timeout) or args.timeout<=0 or not math.isfinite(args.delay) or args.delay<0:
        raise ValueError('Invalid limit, timeout, attempts or delay')
    config=copy.deepcopy(config_override) if config_override is not None else json.loads(args.config.read_text(encoding='utf-8-sig'))
    configuration(config)
    dataset=core.rooted(args.dataset or config['dataset'])
    output=core.rooted(args.output_dir or config['gan_output_dir' if attack_kind=='gan' else 'output_dir'])
    exp=(core.ROOT/'experimenting').resolve()
    if (config['profile']=='experimenting') != output.is_relative_to(exp): raise ValueError('Output must stay in the selected profile')
    source=core.rooted(args.baseline_from) if args.baseline_from else None
    if source==output: raise ValueError('Baseline source must differ from output')
    rows=core.api.load_dataset(dataset,{'clean'});ids={r['sample_id'] for r in rows}
    if not rows or len(ids)!=len(rows): raise ValueError('Dataset needs unique original sample IDs')
    args.selected=([r for r in rows if r['true_label']=='SPAM'][:5]+[r for r in rows if r['true_label']=='NOT_SPAM'][:5]
        if args.pilot else rows[:args.limit] if args.limit else rows)
    if args.pilot and len(args.selected)!=10: raise ValueError('Pilot requires 5 SPAM and 5 NOT_SPAM originals')
    evaluator=config['rewrite_evaluator'];attacker=config['attacker'];prompts=json.loads((core.ROOT/'decision_models/paper_prompts.json').read_text(encoding='utf-8'))
    live=not args.dry_run
    if evaluator.get('rubric_version')!=validity.VERSION: raise ValueError('Unsupported validity rubric version')
    core.validate_endpoint(evaluator,live=live and not args.baseline_only,timeout=args.timeout)
    generator_contract={};engine=None
    if attack_kind=='llm':
        for key in ('temperature','max_tokens'):
            if not isinstance(attacker.get(key),(int,float)) or not math.isfinite(attacker[key]): raise ValueError('Invalid attacker '+key)
        if attacker['temperature']<0 or attacker['max_tokens']<1: raise ValueError('Invalid attacker settings')
        core.validate_endpoint(attacker,attacker=True,live=live and not args.baseline_only,timeout=args.timeout)
        generator_contract=dict(attacker=attacker,prompts=prompts)
    else:
        from gan_engine import gan_contract
        gan=config.get('gan',{})
        if args.gan_resource_dir:gan['resource_dir']=str(args.gan_resource_dir)
        if args.gan_seed is not None:gan['seed']=args.gan_seed
        if int(gan.get('seed',0))<0: raise ValueError('GAN seed must be nonnegative')
        generator_contract=dict(gan=gan_contract(gan,required=live and not args.baseline_only),
            query_policy='generate candidate without defender probe; validity then classify',two_step_extension=True)
    attack_contract=dict(protocol='validity_gated_v1',attack_kind=attack_kind,max_steps=2,
        validity=validity.contract(evaluator),**generator_contract)
    plans=[]
    for name in core.selected_models(config,args.model):
        defender=config['defenders'][name];core.validate_endpoint(defender,live=live,timeout=args.timeout)
        core.defender_payload(defender,'__PREFLIGHT__')
        base=core.fingerprint(dict(dataset=hashlib.sha256(dataset.read_bytes()).hexdigest(),defender=defender,questions=core.api.SYSTEMONE_QUESTION,version=1))
        identities=dict(baseline=base,attack=core.fingerprint({'baseline':base,**attack_contract}))
        ledger=ValidityLedger(output,name,identities,ids)
        if source:
            from gan_pipeline import baseline_source
            baseline_source(source,name,base,ids)
        plans.append((name,defender,identities))
        print(f'{config["profile"]} | {attack_kind} | defender={name} | evaluator=PPLX | max steps=2 | selected={len(args.selected)} | {output/name}',flush=True)
    if args.dry_run: print('DRY RUN: no network, generation, weights loading or output writes.');return 0
    if args.check_server: print('Configuration and resume identities checked; no inference.');return 0
    exit_code=0
    for name,defender,identities in plans:
        with core.file_lock(output/name/'run.lock'):
            ledger=ValidityLedger(output,name,identities,ids);selected=[r['sample_id'] for r in args.selected]
            core.atomic_json(ledger.folder/'run.json',dict(protocol='validity_gated_v1',attack_kind=attack_kind,
                defender=defender,attack_contract=attack_contract,dataset=str(dataset),dataset_rows=len(rows),selected_sample_ids=selected,
                baseline_only=args.baseline_only,identities=identities,experiment_purpose=config.get('experiment_purpose','validity_gated_benchmark'),
                role_limitation='PPLX defender and evaluator share a backend; pilot only' if config['profile']=='experimenting' else 'PPLX evaluator; Clef/d3 defenders'))
            try:
                if source and (source/name/'run.json').exists():
                    from gan_pipeline import baseline_source,import_baseline
                    with core.file_lock(source/name/'run.lock'):
                        previous=baseline_source(source,name,identities['baseline'],ids)
                        if previous:import_baseline(previous,ledger,args.selected,args.retry_errors)
                core.execute_stage(ledger,args.selected,'baseline',defender,attacker,prompts,{},args)
                eligible=[r for r in args.selected if r['true_label']=='SPAM' and ledger.ok(r['sample_id'],'baseline') and ledger.ok(r['sample_id'],'baseline')['label']=='SPAM']
                if not args.baseline_only:
                    start=time.perf_counter()
                    for index,row in enumerate(eligible,1):
                        sid=row['sample_id']
                        for step in (1,2):
                            stage='rewrite'+str(step)
                            if attack_kind=='gan':
                                previous=ledger.get(sid,stage)
                                if not previous or (previous['status']=='ERROR' and args.retry_errors):
                                    if engine is None:
                                        from gan_engine import GanEngine
                                        engine=GanEngine(config.get('gan',{}))
                                    generate_gan(ledger,row,step,engine,args)
                            else:core.execute_stage(ledger,[row],stage,defender,attacker,prompts,{},args)
                            save_individual(ledger,row,step)
                            if not ledger.ok(sid,stage): break
                            validity.evaluate(ledger,row,step,evaluator,args)
                            save_individual(ledger,row,step)
                            decision=ledger.ok(sid,'validate'+str(step))
                            if not decision or decision['label']!='VALID': break
                            core.execute_stage(ledger,[row],'classify'+str(step),defender,attacker,prompts,{},args)
                            save_individual(ledger,row,step)
                            decision=ledger.ok(sid,'classify'+str(step))
                            if not decision or decision['label']!='SPAM': break
                        if index%25==0:report(ledger,args.selected,False)
                        print(core.api.progress(index,len(eligible),time.perf_counter()-start,args.delay),flush=True)
                        if args.delay and index<len(eligible):time.sleep(args.delay)
            except KeyboardInterrupt:
                report(ledger,args.selected,args.baseline_only);ledger.checkpoint('interrupted',selected);return 130
            complete=report(ledger,args.selected,args.baseline_only)
            ledger.checkpoint('complete' if complete else 'incomplete',selected)
            if not complete: exit_code=2
    return exit_code
