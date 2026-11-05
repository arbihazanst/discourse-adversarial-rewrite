#!/usr/bin/env python3
"""Build a fixed CEAS decision-model evaluation package from conversation-provided files.

Requires only Python's standard library. Never infers missing model predictions.
Run: python build_decision_research_handoff.py --source-dir /mnt/data --output-dir /mnt/data/decision_research_handoff
"""
import argparse,csv,hashlib,json,random,shutil,collections,zipfile,io
from pathlib import Path
csv.field_size_limit(1_000_000_000)
MODELS = [
    ('GPT-5.2','LLM_gpt-5.2-chat.csv'),
    ('DeepSeek-V3.1','LLM_DeepSeek-V3.1.csv'),
    ('Qwen-2.5:32B','LLM_qwen2.5_32b.csv'),
    ('GPT-OSS:20B','LLM_gpt_oss_20B.csv'),
    ('Llama-4-Maverick','LLM_Llama-4-Maverick-17B-128E-I.csv'),
    ('Gemma-3:1B','LLM_gemma3_1b.csv'),
]
ARCHIVE_MODELS = [
    ('Qwen-2.5:0.5B','LLM_qwen2.5_0.5b.csv'),
    ('DeepSeek-R1:1.5B','LLM_deepseek_r1_1.5b.csv'),
]
DEFENDERS=[('PPLX Decider Flash v1.2','30002','native_pplx_decider'),('Clef 27B Q8_0','30005','llama.cpp CUDA'),('d3 27B Q8_0','30006','llama.cpp CUDA')]
HEADER_CEAS='CEAS08-'
def csv_read(p,delim=','):
    # Preserve valid UTF-8 exactly while decoding isolated legacy Windows-1252
    # bytes (mixed encodings occur in historical source CSVs). Raw CSVs are also
    # copied into the package completely unchanged for full provenance.
    data=p.read_bytes()
    try:
        text=data.decode('utf-8-sig')
        encoding='utf-8-sig'
    except UnicodeDecodeError:
        dirty=data.decode('utf-8-sig',errors='surrogateescape')
        replacement_count=0
        out=[]
        for ch in dirty:
            code=ord(ch)
            if 0xDC80 <= code <= 0xDCFF:
                raw=bytes([code-0xDC00])
                out.append(raw.decode('cp1252',errors='replace'))
                replacement_count += 1
            else:
                out.append(ch)
        text=''.join(out)
        encoding=f'mixed_utf8_cp1252_fallback_{replacement_count}_bytes'
    rows=list(csv.DictReader(io.StringIO(text),delimiter=delim))
    return rows,encoding
def write_csv(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as fh:
        wr=csv.DictWriter(fh,fieldnames=fields,extrasaction='ignore');wr.writeheader()
        for row in rows:wr.writerow(row)
def sha256_file(p):
    s=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):s.update(b)
    return s.hexdigest()
def label(v):
    x=' '.join((v or '').replace('_',' ').upper().split())
    if x=='SPAM':return 'SPAM'
    if x=='NOT SPAM':return 'NOT_SPAM'
    return 'UNPARSED'
def nonempty(v):return bool(v and v.strip())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',default='/mnt/data');ap.add_argument('--output-dir',default='/mnt/data/decision_research_handoff');args=ap.parse_args()
    base=Path(args.source_dir); dest=Path(args.output_dir);dest.mkdir(parents=True,exist_ok=True)
    ceas,ceas_enc=csv_read(base/'CEAS_08.csv')
    ceas_by_id={i+1:r for i,r in enumerate(ceas)}
    ceas_labels=collections.Counter(str(r['label']).strip() for r in ceas)
    assert len(ceas)==39154,(len(ceas))
    assert ceas_labels['1']==21842 and ceas_labels['0']==17312,ceas_labels
    cohort,enc=csv_read(base/'common_sample_cohorts/six_with_gemma3_1b_source_ids.csv')
    ids=sorted(int(r['CEAS_record_number_1_based']) for r in cohort)
    assert len(ids)==742 and len(set(ids))==742
    assert all(str(ceas_by_id[i]['label'])=='1' for i in ids)
    idset=set(ids)
    crosswalk,_=csv_read(base/'ceas_source_verification/COMPLETE_result_to_CEAS_row_crosswalk.csv')
    file_to_ceas={}
    for row in crosswalk:
        n=row['attack_file']; idx=int(row['attack_record_number_1_based'])
        key=(n,idx)
        if key in file_to_ceas:raise ValueError('Duplicate crosswalk key '+str(key))
        file_to_ceas[key]=int(row['CEAS_record_number_1_based'])
    allmodels=MODELS+ARCHIVE_MODELS
    full_rows=[];selected_rows=[];raw_model_rows={};encodings={};modelstats=[]
    rowid_duplicated=collections.defaultdict(int)
    for name,filename in allmodels:
        raw,encoding=csv_read(base/filename,delim=';')
        raw_model_rows[name]=raw;encodings[filename]=encoding
        match_rows=[]
        seen=set()
        for index,row in enumerate(raw,1):
            ceas_id=file_to_ceas.get((filename,index))
            if ceas_id is None:raise ValueError(f'Missing match for {filename} {index}')
            if str(ceas_by_id[ceas_id]['label'])!='1':raise ValueError('Historical result maps to ham')
            r={
               'source_model':name,'source_csv':filename,'source_result_row_1based':index,
               'ceas_record_id_1based':ceas_id,'ceas_sample_id':f'{HEADER_CEAS}{ceas_id:05}',
               'original_text_as_recorded':row.get('originaltext',''),
               'first_rewrite_text':row.get('firstmodifiedtext',''),
               'first_result_raw':row.get('firstresult',''),
               'first_label_normalized':label(row.get('firstresult')),
               'first_rewrite_nonempty':int(nonempty(row.get('firstmodifiedtext'))),
               'second_rewrite_text':row.get('secondmodified',''),
               'second_result_raw':row.get('secondresult',''),
               'second_label_normalized':label(row.get('secondresult')),
               'second_rewrite_nonempty':int(nonempty(row.get('secondmodified'))),
               'time_sec_recorded':row.get('time_sec',''),
             }
            full_rows.append(r)
            if ceas_id in idset and name in dict(MODELS):
                if ceas_id in seen:
                    rowid_duplicated[name]+=1
                else:
                    match_rows.append(r)
                    seen.add(ceas_id)
        if name in dict(MODELS):
            if len(match_rows)!=742:
                raise ValueError(f'{name} has {len(match_rows)} unique cohort IDs, need 742')
            selected_rows+=match_rows
        stats={
          'model':name,'group':'six_primary_historical' if name in dict(MODELS) else 'archived_excluded_from_primary',
          'total_historical_attack_rows':len(raw),
          'historical_first_spam':sum(label(r['firstresult'])=='SPAM' for r in raw),
          'historical_first_not_spam_raw':sum(label(r['firstresult'])=='NOT_SPAM' for r in raw),
          'historical_first_unparsed':sum(label(r['firstresult'])=='UNPARSED' for r in raw),
          'historical_first_not_spam_with_text':sum(label(r['firstresult'])=='NOT_SPAM' and nonempty(r['firstmodifiedtext']) for r in raw),
          'historical_second_not_spam_after_first_spam_with_text':sum(label(r['firstresult'])=='SPAM' and label(r['secondresult'])=='NOT_SPAM' and nonempty(r['secondmodified']) for r in raw),
          'historical_second_unparsed_after_first_spam':sum(label(r['firstresult'])=='SPAM' and label(r['secondresult'])=='UNPARSED' for r in raw),
          'common742_source_ids':len(match_rows) if name in dict(MODELS) else '',
          'common742_first_not_spam_with_text':sum(r['first_label_normalized']=='NOT_SPAM' and r['first_rewrite_nonempty'] for r in match_rows),
          'common742_second_not_spam_after_first_spam_with_text':sum(r['first_label_normalized']=='SPAM' and r['second_label_normalized']=='NOT_SPAM' and r['second_rewrite_nonempty'] for r in match_rows),
          'common742_first_missing_label':sum(r['first_label_normalized']=='UNPARSED' for r in match_rows),
          'common742_step1_rewrite_nonempty':sum(r['first_rewrite_nonempty'] for r in match_rows),
          'common742_conditional_step2_rewrite_nonempty':sum(r['first_label_normalized']=='SPAM' and r['second_rewrite_nonempty'] for r in match_rows),
        }
        modelstats.append(stats)
    assert len(selected_rows)==742*6,(len(selected_rows))
    fields_selected=['source_model','source_csv','source_result_row_1based','ceas_record_id_1based','ceas_sample_id', 'original_text_as_recorded','first_rewrite_text','first_result_raw','first_label_normalized','first_rewrite_nonempty','second_rewrite_text','second_result_raw','second_label_normalized','second_rewrite_nonempty','time_sec_recorded']
    write_csv(dest/'historical/six_models_common742_recorded_results.csv',fields_selected,selected_rows)
    write_csv(dest/'historical/all_eight_models_recorded_results.csv',fields_selected,full_rows)
    write_csv(dest/'historical/historical_model_outcomes_summary.csv',list(modelstats[0]),modelstats)
    ham_candidates=[];seen_text=set()
    for i,r in enumerate(ceas,1):
        if str(r['label']).strip()!='0':continue
        txt='Subject: '+str(r.get('subject') or '')+'\n\n'+str(r.get('body') or '')
        digest=hashlib.sha256(' '.join(txt.lower().split()).encode('utf-8')).hexdigest()
        if digest in seen_text:continue
        seen_text.add(digest)
        ham_candidates.append(i)
    ham_ids=sorted(random.Random(20261102).sample(ham_candidates,len(ids)))
    def ceas_record(i):
        r=ceas_by_id[i]
        subject=str(r.get('subject') or '')
        body=str(r.get('body') or '')
        return {'sample_id':f'{HEADER_CEAS}{i:05}', 'ceas_record_id_1based':i,
          'label':str(r['label']),'label_name':'SPAM' if str(r['label'])=='1' else 'NOT_SPAM',
          'email_text':'Subject: '+subject+'\n\n'+body,'subject':subject,'body':body,
          'selection':'six_llm_common_source_ids' if str(r['label'])=='1' else 'seed_20261102_random_742_unique_text_ham'}
    clean_spam=[ceas_record(i) for i in ids]; clean_ham=[ceas_record(i) for i in ham_ids]
    cleanfields=['sample_id','ceas_record_id_1based','label','label_name','email_text','subject','body','selection']
    write_csv(dest/'dataset/clean_spam_common742.csv',cleanfields,clean_spam)
    write_csv(dest/'dataset/clean_ham_new742.csv',cleanfields,clean_ham)
    write_csv(dest/'dataset/clean_binary_balanced1484.csv',cleanfields,sorted(clean_spam+clean_ham,key=lambda r:r['ceas_record_id_1based']))
    write_csv(dest/'dataset/common_spam_742_source_ids.csv',['ceas_record_id_1based','sample_id'],[{'ceas_record_id_1based':i,'sample_id':f'{HEADER_CEAS}{i:05}'} for i in ids])
    by_source={(r['ceas_record_id_1based'],r['source_model']):r for r in selected_rows}
    variants=[];all_conditional=[];main=[]
    for i in ids:
        for model,_ in MODELS:
            r=by_source[(i,model)]
            if r['first_rewrite_nonempty']:
                variants.append({'rewrite_id':f'{r["ceas_sample_id"]}:step1:{model}',
                    'sample_id':r['ceas_sample_id'],'ceas_record_id_1based':i,'original_label':'SPAM',
                    'attack_source_model':model,'attack_stage':'step1',
                    'rewrite_text':r['first_rewrite_text'],'original_text':r['original_text_as_recorded'],
                    'historical_source_model_label':r['first_label_normalized'],
                    'historical_source_result_row_1based':r['source_result_row_1based'],
                    'source_csv':r['source_csv']})
            if r['first_label_normalized']=='SPAM' and r['second_rewrite_nonempty']:
                all_conditional.append({'rewrite_id':f'{r["ceas_sample_id"]}:conditional_step2:{model}',
                    'sample_id':r['ceas_sample_id'],'ceas_record_id_1based':i,'original_label':'SPAM',
                    'attack_source_model':model,'attack_stage':'conditional_step2',
                    'rewrite_text':r['second_rewrite_text'],'original_text':r['original_text_as_recorded'],
                    'historical_source_model_label':r['second_label_normalized'],
                    'historical_source_result_row_1based':r['source_result_row_1based'],
                    'source_csv':r['source_csv']})
        # Selection rule frozen independently of attack outcomes, with fallback only for blank text.
        # Hash samples rather than choosing whatever evades a classifier.
        start=int(hashlib.sha256(f'CEAS08:{i}'.encode('utf-8')).hexdigest(),16)%len(MODELS)
        eligible=[MODELS[(start+j)%len(MODELS)][0] for j in range(len(MODELS))]
        source=next((m for m in eligible if by_source[(i,m)]['first_rewrite_nonempty']),None)
        if source is None:
            raise ValueError(f'Sample {i} has no nonempty first rewrite in any of the six models; cannot create 742-row primary')
        r=by_source[(i,source)]
        main.append({'rewrite_id':f'{r["ceas_sample_id"]}:primary_step1:{source}',
            'sample_id':r['ceas_sample_id'],'ceas_record_id_1based':i,'original_label':'SPAM',
            'attack_source_model':source,'attack_stage':'step1',
            'rewrite_text':r['first_rewrite_text'],'original_text':ceas_record(i)['email_text'],
            'historical_source_model_label':r['first_label_normalized'],
            'historical_source_result_row_1based':r['source_result_row_1based'],
            'source_csv':r['source_csv']})
    rewritefields=['rewrite_id','sample_id','ceas_record_id_1based','original_label','attack_source_model','attack_stage','rewrite_text','original_text','historical_source_model_label','historical_source_result_row_1based','source_csv']
    write_csv(dest/'dataset/frozen_primary_step1_one_per_spam742.csv',rewritefields,main)
    write_csv(dest/'dataset/frozen_all_six_step1_nonempty.csv',rewritefields,variants)
    write_csv(dest/'dataset/frozen_all_six_conditional_step2_nonempty.csv',rewritefields,all_conditional)
    write_csv(dest/'dataset/frozen_all_six_steps_pooled.csv',rewritefields,variants+all_conditional)
    # A single long-form, directly runnable benchmark input manifest: every
    # decision defender receives exactly these 1,484 clean + 742 rewrite rows.
    input_rows=[]
    for r in clean_spam+clean_ham:
        input_rows.append({'input_id':r['sample_id']+':clean', 'sample_id':r['sample_id'],
            'kind':'clean','true_label':r['label_name'],'email_text':r['email_text'],
            'source_attack_model':'', 'original_ceas_id':r['ceas_record_id_1based']})
    for r in main:
        input_rows.append({'input_id':r['rewrite_id'], 'sample_id':r['sample_id'],
            'kind':'frozen_primary_step1','true_label':'SPAM','email_text':r['rewrite_text'],
            'source_attack_model':r['attack_source_model'], 'original_ceas_id':r['ceas_record_id_1based']})
    input_fields=['input_id','sample_id','kind','true_label','email_text','source_attack_model','original_ceas_id']
    write_csv(dest/'dataset/benchmark_inputs_clean1484_plus_rewrite742.csv',input_fields,input_rows)
    assert len(input_rows)==2226 and len({r['input_id'] for r in input_rows})==2226
    # Templates deliberately have no model result content.
    template=[]
    for defender,port,backend in DEFENDERS:
        for r in clean_spam+clean_ham:
            template.append({'defender':defender,'api_port':port,'engine':backend,'item_type':'clean',
                'item_id':r['sample_id'],'sample_id':r['sample_id'],'true_label':r['label_name'],
                'predicted_label':'','p_spam':'','p_not_spam':'','latency_ms':'','status':'NOT_RUN'})
        for r in main:
            template.append({'defender':defender,'api_port':port,'engine':backend,'item_type':'frozen_primary_step1',
                'item_id':r['rewrite_id'],'sample_id':r['sample_id'],'true_label':'SPAM',
                'predicted_label':'','p_spam':'','p_not_spam':'','latency_ms':'','status':'NOT_RUN'})
    write_csv(dest/'decision_models/decision_predictions_template.csv',list(template[0]),template)
    decision_cfg={'models':[{'name':n,'host_port':port,'engine':eng,'status':'not_yet_evaluated_on_this_CEAS_package'} for n,port,eng in DEFENDERS],
        'label_options':['SPAM','NOT_SPAM'],'note':'PPLX native request adapter differs from llama.cpp systemone; benchmark app must normalize response shapes without changing semantic criteria.',
        'suggested_systemone':{'questions':{'spam':{'type':'choice','instructions':'Classify the content of the email as SPAM or NOT_SPAM.','criteria':{'SPAM':'Unsolicited advertising, phishing, scams, fraudulent offers, or deceptive email.','NOT_SPAM':'Legitimate personal, business, or informational correspondence.'}}}}}
    # Preserve the shipped deployment mapping and include the runnable pipeline
    # when rebuilding into a fresh directory.
    bundled_root = Path(__file__).resolve().parent
    bundled_config = bundled_root/'decision_models/decision_model_config.json'
    if bundled_config.exists():
        decision_cfg = json.loads(bundled_config.read_text(encoding='utf-8-sig'))
    (dest/'decision_models/decision_model_config.json').write_text(json.dumps(decision_cfg,ensure_ascii=False,indent=2),encoding='utf8')
    for script in (bundled_root/'scripts').glob('*'):
        if script.is_file() and script.suffix in ('.py', '.sh', '.md', '.example'):
            target = dest/'scripts'/script.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if script.resolve() != target.resolve():
                shutil.copy2(script,target)
    for name,file in allmodels:
        sub='historical/raw_six_models' if name in dict(MODELS) else 'historical/raw_excluded_archived'
        (dest/sub).mkdir(parents=True,exist_ok=True)
        shutil.copy2(base/file,dest/sub/file)
    ganfiles=sorted(base.glob('GAN_*.csv'))
    (dest/'historical/raw_GAN_archive_only').mkdir(parents=True,exist_ok=True)
    for p in ganfiles:shutil.copy2(p,dest/'historical/raw_GAN_archive_only'/p.name)
    # Preserve audit context; don't assert GAN ASR is verified.
    for p in ['manuscript_reconciliation.csv','llm_missing_outcomes_audit.csv','gan_label_rule_audit.csv','llm_observed_outcome_states.csv','source_outcome_counts.csv','join_audit.csv','length_truncation_rates.csv']:
        (dest/'historical/audits').mkdir(parents=True,exist_ok=True)
        shutil.copy2(base/p,dest/'historical/audits'/p)
    for p in ['six_with_gemma3_1b_source_ids.csv','cohort_sizes.csv','first_rewrite_identity_by_cohort.csv','first_rewrite_identity_pairwise.csv']:
        (dest/'lineage').mkdir(parents=True,exist_ok=True)
        shutil.copy2(base/'common_sample_cohorts'/p,dest/'lineage'/p)
    shutil.copy2(base/'ceas_source_verification/COMPLETE_result_to_CEAS_row_crosswalk.csv',dest/'lineage/CEAS_attack_results_full_crosswalk.csv')
    shutil.copy2(base/'ceas_source_verification/FINDINGS.md',dest/'lineage/CEAS_lineage_findings.md')
    primary_src=collections.Counter(r['attack_source_model'] for r in main)
    summaries={
      'input_CEAS_total_records':len(ceas),'input_CEAS_spam_records':ceas_labels['1'],'input_CEAS_ham_records':ceas_labels['0'],
      'primary_historical_LLM_models':len(MODELS),'excluded_archived_LLM_models':len(ARCHIVE_MODELS),
      'common_spam_CEAS_ids':len(ids),'new_ham_CEAS_ids':len(ham_ids),'clean_balanced_binary_rows':len(ids)+len(ham_ids),
      'common742_history_model_rows':len(selected_rows),'all_historical_LLM_attack_rows':len(full_rows),
      'primary_frozen_step1_rows':len(main),'fixed_benchmark_inputs':len(input_rows),'all_six_nonempty_step1_rows':len(variants),
      'all_six_nonempty_conditional_step2_rows':len(all_conditional),'all_six_step_pooled_rows':len(variants+all_conditional),
      'new_decision_prediction_template_rows':len(template),'decision_model_completed_predictions':0,
      'primary_frozen_step1_attack_source_distribution':dict(primary_src),
      'duplicate_same_source_id_rows_dropped_for_common742_comparison':dict(rowid_duplicated),
      'source_encodings':encodings,
      'ham_selection':'Deterministic random sample of 742 unique normalized-text CEAS label=0 records, seed 20261102; not the original paper ham sample',
      'primary_rewrite_selection':'SHA256(CEAS08:<source_id>) integer modulo six model positions; cyclic fallback ONLY if Step1 rewrite empty, never choose by evasion outcome',
    }
    (dest/'COUNTS.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8')
    table='\n'.join(f"| {r['model']} | {r['total_historical_attack_rows']} | {r['historical_first_not_spam_with_text']} | {r['historical_second_not_spam_after_first_spam_with_text']} | {r['common742_first_not_spam_with_text'] if r['common742_first_not_spam_with_text']!='' else '—'} |" for r in modelstats)
    readme=f'''# Decision-model spam classification research handoff — 2026-11-02

## What is included

This package was assembled from the previously supplied `CEAS_08.csv`, 8 original LLM CSVs, 8 GAN CSVs, and the earlier line-by-line CEAS source crosswalk. All historical model outputs are **recorded observations**, not new re-runs. No decision-model classification outputs were provided; these are left as `NOT_RUN`.

### Main datasets

- `dataset/clean_binary_balanced1484.csv`: **1,484 labeled clean emails**: 742 shared CEAS spam emails plus 742 newly selected CEAS legitimate emails. `label=1` means SPAM; `label=0` means NOT_SPAM. Original CEAS subject/body preserved (not victim-specific corrupted text). The 742 ham emails were not shown to be the manuscript's original 1,500 ham emails.
- `dataset/clean_spam_common742.csv`, `dataset/clean_ham_new742.csv`: class-specific subsets.
- `dataset/benchmark_inputs_clean1484_plus_rewrite742.csv`: **2,226 fixed classifier input records** in one machine-readable table, covering all clean emails and the primary frozen attacks.
- `dataset/frozen_primary_step1_one_per_spam742.csv`: **742 nonempty, preselected Step-1 rewrites**; exactly one per spam source. This is the practical identical-input transfer set for *every* new defender. Selection uses SHA256 of source ID, rotating across the six historical source models; if its Step-1 rewrite is blank, use the next source model in fixed cyclic order. Does **not** choose attacks based on whether they fooled any victim.
- `dataset/frozen_all_six_step1_nonempty.csv`: all nonempty Step-1 rewrites of the 742 shared original emails (may contain different variants for the same email).
- `dataset/frozen_all_six_conditional_step2_nonempty.csv`: observed nonempty Step-2 rewrites **only for source records where Step-1 prediction was SPAM**. A blank step or unknown label is not imputed as a failed attempt.
- `dataset/frozen_all_six_steps_pooled.csv`: fixed pooled rewrite options for broader transferability checks; **do not treat multiple variants from the same original email as independent observations**.

### Historical results

- `historical/six_models_common742_recorded_results.csv`: 6 × 742 unique CEAS source IDs with original Step-1/Step-2 rewrite text and raw outcomes. For duplicate CEAS IDs within a model, the first recorded occurrence is retained, consistent with earlier overlap audit; full originals are preserved.
- `historical/all_eight_models_recorded_results.csv`: full normalized-UTF-8 representation of eight LLM CSVs with source IDs, original text, stage labels, nonempty flags. Original raw files also included.
- `historical/historical_model_outcomes_summary.csv`: descriptive counts. **NOT validated adversarial success rates**; missing/unparsed outcomes are retained, semantic intent is not independently human-validated.
- `historical/raw_six_models/`: exact six original LLM CSVs, unmodified.
- `historical/raw_excluded_archived/`: the excluded DeepSeek-R1:1.5B and Qwen-2.5:0.5B original LLM CSVs.
- `historical/raw_GAN_archive_only/`: eight original GAN CSVs archived for provenance **not** suitable for confirmed ASR comparison because historic label mapping is unresolved.
- `historical/audits/`: earlier CSV reconciliation reports.

### Lineage

- `lineage/CEAS_attack_results_full_crosswalk.csv`: exact historical attack row -> original CEAS source ID with match method and notes. Prior manual normalization involved **26 near-match records**; some CEAS texts map to multiple identical source rows. Treat IDs as deterministic provenance mappings rather than byte-identical evidence in those cases.
- `lineage/six_with_gemma3_1b_source_ids.csv`: 742 common CEAS source IDs.

### Decision models — new results not yet available

`decision_models/decision_predictions_template.csv` has {len(template)} rows = 3 decision models × (1,484 clean + 742 frozen attack texts). `predicted_label`, `p_spam`, `p_not_spam` and `latency_ms` are empty, `status=NOT_RUN`. Populate *only* by performing actual inference.

| Decision model | Port (GPU1 host) | Native engine |
|---|---|---|
| PPLX Decider Flash v1.2 | 30002 | pplx_decider native service |
| Clef 27B Q8_0 | 30005 | llama.cpp CUDA |
| d3 27B Q8_0 | 30006 | llama.cpp CUDA |

Docker start/stop is one model at a time. Your Open WebUI API routing may differ; the package does **not** assume a proven working remote API for each model.

### Historical descriptive counts

First and conditional Second columns are observed `NOT_SPAM` labels **with nonempty rewrite text**; these are label-flip candidates, not human-validated valid attacks.

| Source model | Original LLM rows | First-stage NOT_SPAM+text | Conditional second-stage NOT_SPAM+text | First-stage NOT_SPAM+text in common742 |
|---|---:|---:|---:|---:|
{table}

## Required methodological caveats

1. The historic per-victim attack pools are not identical (guardrail blocking and replacement sampling), although their output counts were targeted to be similar.
2. The 742 common IDs control *original email identity*. Their recorded rewrite texts were generally different across the six victim models.
3. The frozen rewrite corpus permits identical-input comparison for **new** defenders. The historical six generative LLM predictions do **not** automatically correspond to these same 742 frozen rewritten texts. **Re-evaluate those LLMs on the frozen corpus** for a fair six-versus-three head-to-head robustness result.
4. The attack results contain source spam emails only. New ham selection enables clean binary classification but is **not** identical to the historical clean baseline. Historical clean accuracy cannot be reconstructed from attack logs alone.
5. Historical GAN success labels remain ambiguous; do not claim general GAN inferiority based on these files.
6. No new decision model results or semantic-validity human assessments are present. Decision probabilities should be checked for their actual field shapes and calibration. Do not treat similarity scores as validated spam-intent preservation.
7. The same CEAS original email can appear multiple times due to duplicates; full provenance is included. Do not silently count multiple rewrites as statistically independent samples.
8. Clean ham emails are a **new fixed sample** selected from the uploaded CEAS label-0 pool; confirm labels/data quality before publication.
9. Report backend (pplx native vs llama.cpp), checkpoint quantization (Q8 for Clef/d3), exact revisions and GPU device.

## How to rebuild

Place the original files and analysis crosswalk in one directory, then:

```bash
python build_decision_research_handoff.py --source-dir /path/to/source --output-dir ./decision_research_handoff
```

`COUNTS.json` and `MANIFEST.json` provide inspected counts and SHA256 hashes.
'''
    if (dest/'scripts/README.md').exists():
        readme += '\n\n## Additional benchmark scripts\n\nRun `python scripts/run_decision_benchmark.py --help` and read `scripts/README.md` for the Open WebUI and direct host-port modes (PPLX 30002, Clef 30005, d3 30006). Native PPLX schema requires verification.\n'
    (dest/'README.md').write_text(readme,encoding='utf-8')
    source_script=Path(__file__)
    if source_script.resolve() != (dest/'build_decision_research_handoff.py').resolve():shutil.copy2(source_script,dest/'build_decision_research_handoff.py')
    manifest=[]
    for path in sorted(dest.rglob('*')):
        if path.is_file() and path.name!='MANIFEST.json':
            manifest.append({'relative_path':str(path.relative_to(dest)),'bytes':path.stat().st_size,'sha256':sha256_file(path)})
    (dest/'MANIFEST.json').write_text(json.dumps({'generated':'2026-11-02','source':'User-provided CEAS_08.csv, original LLM/GAN CSVs, established crosswalk','files':manifest},indent=2),encoding='utf-8')
    zip_out=base/'decision_models_common_ceas_research_2026-11-02.zip'
    with zipfile.ZipFile(zip_out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,arcname='decision_research_handoff/'+str(p.relative_to(dest)))
    print(json.dumps({'zip':str(zip_out),'zip_MB':round(zip_out.stat().st_size/1024**2,2),'counts':summaries},indent=2))
if __name__=='__main__':main()
