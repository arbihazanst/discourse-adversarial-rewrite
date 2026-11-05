"""Audit the newly supplied baseline; preserve original data and inference outputs."""
import collections
import csv
import hashlib
import json
import runpy
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OUT=HERE/'baseline_recovery'
OUT.mkdir(exist_ok=True)
INPUT=Path(r'C:/KERJA DANI/vscode/OPEN ATTACK/OpenAttack-Share/CEAS_08_gpt_oss_latest_transparent_20260220_185010.csv')
h=runpy.run_path(str(HERE/'match_ceas.py'))
read,ws,digest=h['read_csv'],h['ws'],h['digest']


def save(name,rows,fields=None):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


baseline=read(INPUT)
source=read(HERE/'CEAS_08.csv')
source_text=['Subject: '+r['subject']+'\n\n'+r['body'] for r in source]
indexes={k:collections.defaultdict(list) for k in ['exact','whitespace_case','missing_subject_nan']}
for i,(r,text) in enumerate(zip(source,source_text),1):
    indexes['exact'][digest(text)].append(i)
    indexes['whitespace_case'][digest(ws(text))].append(i)
    if not r['subject']:
        indexes['missing_subject_nan'][digest(ws('Subject: nan\n\n'+r['body']))].append(i)
manifest=[]
baseline_by_recorded=collections.defaultdict(list)
baseline_by_source=collections.defaultdict(list)
for n,r in enumerate(baseline,1):
    ids=[];kind='unresolved'
    for method,key in [('exact',digest(r['text'])),('whitespace_case',digest(ws(r['text']))),
                       ('missing_subject_nan',digest(ws(r['text'])))]:
        ids=indexes[method].get(key,[])
        if ids:kind=method;break
    baseline_by_recorded[ws(r['text'])].append(n)
    if ids:
        # Equal normalized source text may occur at several CEAS record IDs.
        assert len({ws(source_text[i-1]) for i in ids})==1
        baseline_by_source[ws(source_text[ids[0]-1])].append(n)
    manifest.append({'baseline_row_1based':n,'baseline_input_id':f'baseline-gptoss-{n:04}',
        'true_label':r['true_label'],'historical_predicted_label':r['predicted_label'],
        'match_type':kind,'candidate_ceas_ids':'|'.join(map(str,ids)),
        'candidate_count':len(ids),'unique_ceas_id':ids[0] if len(ids)==1 else '',
        'label_agrees_with_all_source_candidates':int(bool(ids) and all(source[i-1]['label']==r['true_label'] for i in ids)),
        'original_input_sha256':digest(r['text']),'original_input_text':r['text']})
save('baseline_3000_manifest.csv',manifest)
duplicates=[{'baseline_rows':'|'.join(map(str,ns)), 'occurrences':len(ns),
             'labels':'|'.join(baseline[i-1]['true_label'] for i in ns),
             'predictions':'|'.join(baseline[i-1]['predicted_label'] for i in ns)}
            for ns in baseline_by_recorded.values() if len(ns)>1]
save('duplicate_baseline_inputs.csv',duplicates)
historical=read(ROOT/'historical/all_eight_models_recorded_results.csv')
prior={(r['attack_file'],int(r['attack_record_number_1_based'])):r
       for r in read(HERE/'historical_matches.csv')}
overlap=[]
for r in historical:
    direct=baseline_by_recorded.get(ws(r['original_text_as_recorded']),[])
    sid=int(r['ceas_record_id_1based'])
    linked=baseline_by_source.get(ws(source_text[sid-1]),[])
    previous=prior[(r['source_csv'],int(r['source_result_row_1based']))]
    overlap.append({'model':r['source_model'],'source_csv':r['source_csv'],
        'attack_row_1based':r['source_result_row_1based'],'historical_ceas_id':sid,
        'direct_text_baseline_rows':'|'.join(map(str,direct)),
        'historical_source_link_baseline_rows':'|'.join(map(str,linked)),
        'historical_source_link_status':previous['identity_status'],
        'baseline_predictions_for_linked_source':'|'.join(sorted({baseline[n-1]['predicted_label'] for n in linked})),
        'baseline_labels_for_linked_source':'|'.join(sorted({baseline[n-1]['true_label'] for n in linked})),
        'step1_label':r['first_label_normalized'],
        'step1_nonempty':r['first_rewrite_nonempty'],
        'step2_label':r['second_label_normalized'],
        'step2_nonempty':r['second_rewrite_nonempty']})
save('historical_attack_overlap.csv',overlap)
by_model=[]
for model in sorted({r['model'] for r in overlap}):
    rows=[r for r in overlap if r['model']==model]
    by_model.append({'model':model,'historical_rows':len(rows),
        'recorded_text_found':sum(bool(r['direct_text_baseline_rows']) for r in rows),
        'historically_linked_source_found':sum(bool(r['historical_source_link_baseline_rows']) for r in rows),
        'historically_linked_source_missing':sum(not r['historical_source_link_baseline_rows'] for r in rows),
        'linked_to_unambiguous_gptoss_baseline_predicted_0':sum(r['baseline_predictions_for_linked_source']=='0' for r in rows) if model=='GPT-OSS:20B' else '',
        'historical_unique_source_ids':len({r['historical_ceas_id'] for r in rows})})
save('overlap_by_model.csv',by_model)
gpt_issues=[r for r in overlap if r['model']=='GPT-OSS:20B' and
            (not r['historical_source_link_baseline_rows'] or r['baseline_predictions_for_linked_source']!='1')]
save('gptoss_attack_pool_discrepancies.csv',gpt_issues,list(overlap[0]))
current=read(ROOT/'dataset/clean_binary_balanced1484.csv')
current_overlap=[]
for r in current:
    ns=baseline_by_source.get(ws(r['email_text']),[])
    current_overlap.append({'sample_id':r['sample_id'],'true_label':r['label'],
                            'baseline_rows':'|'.join(map(str,ns))})
save('current_benchmark_overlap.csv',current_overlap)
conf=collections.Counter((r['true_label'],r['predicted_label']) for r in baseline)
tp,tn,fp,fn=[conf[x] for x in [('1','1'),('0','0'),('0','1'),('1','0')]]
gptsource={ws(source_text[int(r['ceas_record_id_1based'])-1]) for r in historical if r['source_model']=='GPT-OSS:20B'}
tpkeys={k for k,ns in baseline_by_source.items() if any(baseline[n-1]['true_label']=='1' and baseline[n-1]['predicted_label']=='1' for n in ns)}
missing_tp=[{'baseline_rows':'|'.join(map(str,baseline_by_source[k])),
             'source_normalized_sha256':digest(k)} for k in sorted(tpkeys-gptsource)]
save('gptoss_baseline_tp_without_historical_attack.csv',missing_tp,['baseline_rows','source_normalized_sha256'])
(OUT/INPUT.name).write_bytes(INPUT.read_bytes())
summary={'source_file':str(INPUT),'source_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    'baseline_rows':len(baseline),'true_label_counts':dict(collections.Counter(r['true_label'] for r in baseline)),
    'match_counts':dict(collections.Counter(r['match_type'] for r in manifest)),
    'source_label_agreement_rows':sum(r['label_agrees_with_all_source_candidates'] for r in manifest),
    'ambiguous_ceas_candidate_rows':sum(r['candidate_count']>1 for r in manifest),
    'unique_normalized_recorded_inputs':len(baseline_by_recorded),
    'duplicate_input_groups':len(duplicates),'extra_duplicate_rows':len(baseline)-len(baseline_by_recorded),
    'TP':tp,'TN':tn,'FP':fp,'FN':fn,'accuracy':(tp+tn)/len(baseline),
    'spam_recall':tp/(tp+fn),'spam_precision':tp/(tp+fp),
    'spam_f1':2*tp/(2*tp+fp+fn),
    'historical_overlap':by_model,
    'gptoss_baseline_unique_tp_sources_without_historical_attack':len(missing_tp),
    'current_clean_spam_found':sum(r['true_label']=='1' and bool(r['baseline_rows']) for r in current_overlap),
    'current_clean_ham_found':sum(r['true_label']=='0' and bool(r['baseline_rows']) for r in current_overlap),
    'same_3000_baseline_for_all_historical_models_proven':False,
    'input_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [INPUT,HERE/'CEAS_08.csv',ROOT/'historical/all_eight_models_recorded_results.csv',
         HERE/'historical_matches.csv',ROOT/'dataset/clean_binary_balanced1484.csv']}}
assert len(manifest)==3000
assert summary['source_label_agreement_rows']==3000
assert tp+tn+fp+fn==3000
assert (OUT/INPUT.name).read_bytes()==INPUT.read_bytes()
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='input_hashes'},indent=2))
