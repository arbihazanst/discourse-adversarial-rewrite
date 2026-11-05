"""Count available historical rewrite candidates for the recovered baseline.

Availability only: no selection based on attack success, no semantic validation,
no model calls, and no modification of benchmark/pipeline files.
"""
import collections
import csv
import json
import runpy
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
read=runpy.run_path(str(HERE/'match_ceas.py'))['read_csv']
baseline=read(HERE/'baseline_recovery/baseline_3000_manifest.csv')
historical=read(ROOT/'historical/all_eight_models_recorded_results.csv')
matches={(r['attack_file'],int(r['attack_record_number_1_based'])):r
         for r in read(HERE/'historical_matches.csv')}
excluded={'Qwen-2.5:0.5B','DeepSeek-R1:1.5B'}
strong=collections.defaultdict(list)
review=collections.defaultdict(list)
for row in historical:
    if row['source_model'] in excluded or not row['first_rewrite_text'].strip():continue
    m=matches[(row['source_csv'],int(row['source_result_row_1based']))]
    if m['match_type'] in {'exact','line_endings','whitespace_case','typography'}:
        for sid in m['candidate_ceas_ids'].split('|'):
            if sid:strong[sid].append(row)
    else:
        review[row['ceas_record_id_1based']].append(row)
rows=[]
for b in baseline:
    if b['true_label']!='1':continue
    ids=b['candidate_ceas_ids'].split('|')
    found={(r['source_csv'],r['source_result_row_1based']):r for sid in ids for r in strong[sid]}
    uncertain={(r['source_csv'],r['source_result_row_1based']):r for sid in ids for r in review[sid]}
    rows.append({'baseline_row_1based':b['baseline_row_1based'],
        'candidate_ceas_ids':b['candidate_ceas_ids'],
        'nonempty_step1_candidates_fulltext_source_match':len(found),
        'nonempty_step1_candidates_review_required':len(uncertain),
        'historical_defenders':'|'.join(sorted({r['source_model'] for r in found.values()})),
        'candidate_chains_with_conditional_step2':sum(r['first_label_normalized']=='SPAM' and bool(r['second_rewrite_text'].strip()) for r in found.values()),
        'meaning_preservation_validated':False})
with (HERE/'baseline_recovery/rewrite_availability.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'baseline_spam_rows':len(rows),
    'with_nonempty_step1_fulltext_source_match':sum(r['nonempty_step1_candidates_fulltext_source_match']>0 for r in rows),
    'only_review_required_step1_candidates':sum(r['nonempty_step1_candidates_fulltext_source_match']==0 and r['nonempty_step1_candidates_review_required']>0 for r in rows),
    'without_step1_candidate_in_six_model_archive':sum(r['nonempty_step1_candidates_fulltext_source_match']==0 and r['nonempty_step1_candidates_review_required']==0 for r in rows),
    'with_at_least_one_historical_conditional_step2_chain':sum(r['candidate_chains_with_conditional_step2']>0 for r in rows),
    'scope':'Six primary archived historical defenders; excludes tiny-model archives and new pilot CSVs. Counts input rows including baseline duplicates. Nonempty text is not semantic validity or verified prompt/checkpoint provenance.'}
assert sum(summary[k] for k in ['with_nonempty_step1_fulltext_source_match','only_review_required_step1_candidates','without_step1_candidate_in_six_model_archive'])==1500
(HERE/'baseline_recovery/rewrite_availability_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
