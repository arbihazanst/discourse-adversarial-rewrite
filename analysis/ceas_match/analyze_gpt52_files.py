"""Audit the user-supplied GPT-5.2 raw and combined files without modifying them."""
import collections
import csv
import hashlib
import io
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT.parent/'adversarialattack'
OUT = HERE/'gpt52_file_audit'
OUT.mkdir(exist_ok=True)
helpers = runpy.run_path(str(HERE/'match_ceas.py'))
read_csv = helpers['read_csv']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


raw_path = OLD/'results/LLM_gpt-5.2-chat.csv'
combined_path = OLD/'combined_with_analysis/combined_with_analysis_gpt-5.2-chat.csv'
gan_path = OLD/'results/GAN_gpt-5.2-chat.csv'
archived = ROOT/'historical/raw_six_models/LLM_gpt-5.2-chat.csv'
raw = read_csv(raw_path,';')
combined = read_csv(combined_path,'|')
# Reproduce the old merge.py decoding solely for source-key verification.
raw_latin = list(csv.DictReader(io.StringIO(raw_path.read_bytes().decode('latin1')),delimiter=';'))
gan_latin = list(csv.DictReader(io.StringIO(gan_path.read_bytes().decode('latin1')),delimiter=';'))
raw_keys = collections.defaultdict(list)
for i,row in enumerate(raw_latin,1):
    raw_keys[row['originaltext'].strip()].append(i)
gan_keys=collections.Counter(r['text'].strip() for r in gan_latin)
combined_keys=collections.Counter(r['merge_text'] for r in combined)
expected=collections.Counter({k:len(v)*gan_keys[k] for k,v in raw_keys.items() if gan_keys[k]})
matches={int(r['attack_record_number_1_based']):r for r in read_csv(HERE/'historical_matches.csv')
         if r['attack_file']=='LLM_gpt-5.2-chat.csv'}
source=read_csv(HERE/'CEAS_08.csv')
row_map=[]
for i,row in enumerate(raw,1):
    m=matches[i]
    row_map.append({'raw_result_row_1based':i,'old_ceas_id':m['old_ceas_id'],
        'source_label':source[int(m['old_ceas_id'])-1]['label'],
        'fresh_match_type':m['match_type'],'identity_status':m['identity_status'],
        'candidate_ceas_ids':m['candidate_ceas_ids'],
        'old_id_in_fresh_candidates':m['old_id_in_candidates'],
        'present_in_combined':int(raw_latin[i-1]['originaltext'].strip() in combined_keys),
        'step1_label':row['firstresult'],'step1_text_nonempty':int(bool(row['firstmodifiedtext'].strip())),
        'step2_label':row['secondresult'],'step2_text_nonempty':int(bool(row['secondmodified'].strip())),
        'original_text_sha256':helpers['digest'](row['originaltext'])})
combined_map=[]
text_columns=['originaltext','firstmodifiedtext','secondmodified']
for i,row in enumerate(combined,1):
    candidates=raw_keys.get(row['merge_text'],[])
    full_matches=[n for n in candidates if all(row['llm_'+col]==raw_latin[n-1][col]
        for col in text_columns+['firstresult','secondresult'])]
    combined_map.append({'combined_row_1based':i,'raw_row_candidates':'|'.join(map(str,candidates)),
        'all_llm_texts_and_labels_match_raw_latin1':int(bool(full_matches)),
        'mapped_ceas_ids':'|'.join(sorted({matches[n]['old_ceas_id'] for n in full_matches},key=int)),
        'gan_original_equals_llm_original_after_strip':int(row['gan_text'].strip()==row['llm_originaltext'].strip()),
        'original_differs_from_correctly_decoded_raw':int(bool(full_matches) and
            row['llm_originaltext']!=raw[full_matches[0]-1]['originaltext'])})
by_id=collections.defaultdict(list)
for row in row_map:by_id[int(row['old_ceas_id'])].append(row)
source_subset=[]
for sid,rows in sorted(by_id.items()):
    src=source[sid-1]
    source_subset.append({'ceas_record_id_1based':sid,'sample_id':f'CEAS08-{sid:05}',
        'ground_truth_label':src['label'],'historical_result_rows':'|'.join(str(r['raw_result_row_1based']) for r in rows),
        'historical_occurrences':len(rows),
        'identity_statuses':'|'.join(sorted({r['identity_status'] for r in rows})),
        'source_id_policy':'previous_mapping_preserved; inspect row_mapping.csv for ambiguity',
        'subject':src['subject'],'body':src['body'],'email_text':'Subject: '+src['subject']+'\n\n'+src['body']})
dropped=[r for r in row_map if not r['present_in_combined']]
save('row_mapping.csv',row_map)
save('combined_to_raw_mapping.csv',combined_map)
save('dropped_by_inner_join.csv',dropped)
save('historically_mapped_spam_sources.csv',source_subset)
summary={
    'raw_rows':len(raw),'combined_rows':len(combined),'raw_same_bytes_as_previously_audited':sha(raw_path)==sha(archived),
    'combined_keys_equal_reproduced_inner_join':combined_keys==expected,
    'combined_unknown_original_keys':len(set(combined_keys)-set(raw_keys)),
    'combined_rows_llm_texts_labels_reproduced':sum(r['all_llm_texts_and_labels_match_raw_latin1'] for r in combined_map),
    'combined_originals_with_encoding_differences':sum(r['original_differs_from_correctly_decoded_raw'] for r in combined_map),
    'distinct_historically_mapped_ceas_ids':len(by_id),
    'extra_occurrences_of_mapped_source_ids':len(raw)-len(by_id),
    'all_mapped_source_labels':dict(collections.Counter(r['source_label'] for r in row_map)),
    'fresh_match_counts':dict(collections.Counter(r['fresh_match_type'] for r in row_map)),
    'identity_status_counts':dict(collections.Counter(r['identity_status'] for r in row_map)),
    'dropped_rows':len(dropped),'dropped_raw_row_numbers':[r['raw_result_row_1based'] for r in dropped],
    'dropped_step1_labels':dict(collections.Counter(r['step1_label'] for r in dropped)),
    'first_result_counts':dict(collections.Counter(r['firstresult'] for r in raw)),
    'second_result_counts':dict(collections.Counter(r['secondresult'] for r in raw)),
    'blank_first_result_rows':[i for i,r in enumerate(raw,1) if not r['firstresult'].strip()],
    'blank_step1_texts':sum(not r['firstmodifiedtext'].strip() for r in raw),
    'conditional_step2_not_spam':sum(r['firstresult']=='SPAM' and r['secondresult']=='NOT SPAM' and bool(r['secondmodified'].strip()) for r in raw),
    'input_sha256':{str(p):sha(p) for p in [raw_path,combined_path,gan_path,archived,OLD/'merge.py',HERE/'CEAS_08.csv',HERE/'historical_matches.csv']},
    'original_3000_baseline_recovered':False,
}
assert summary['combined_keys_equal_reproduced_inner_join']
assert summary['combined_rows_llm_texts_labels_reproduced']==len(combined)
assert len(raw)==len(combined)+len(dropped)
assert sum(r['historical_occurrences'] for r in source_subset)==len(raw)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2))
