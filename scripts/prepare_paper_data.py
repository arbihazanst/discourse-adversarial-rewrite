"""Materialize the recovered paper baseline and an outcome-independent Step-1 bank."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = ROOT/'analysis/ceas_match/baseline_recovery'
    manifest = read(audit/'baseline_3000_manifest.csv')
    source = audit/'CEAS_08_gpt_oss_latest_transparent_20260220_185010.csv'
    original = read(source)
    expected = json.loads((audit/'summary.json').read_text())['source_sha256']
    if sha(source) != expected or len(original) != 3000 or len(manifest) != 3000:
        raise ValueError('Recovered baseline provenance/size mismatch')
    historical = read(ROOT/'historical/all_eight_models_recorded_results.csv')
    matches = {(r['attack_file'], r['attack_record_number_1_based']): r
               for r in read(ROOT/'analysis/ceas_match/historical_matches.csv')}
    by_source = {}
    excluded = {'Qwen-2.5:0.5B', 'DeepSeek-R1:1.5B'}
    # File/record order is fixed BEFORE considering any classifier outcome.
    for r in sorted(historical, key=lambda x: (x['source_csv'], int(x['source_result_row_1based']))):
        if r['source_model'] in excluded or not r['first_rewrite_text'].strip():
            continue
        m = matches[(r['source_csv'], r['source_result_row_1based'])]
        if m['match_type'] not in {'exact', 'line_endings', 'whitespace_case', 'typography'}:
            continue
        for sid in m['candidate_ceas_ids'].split('|'):
            if sid:
                by_source.setdefault(sid, []).append((r, m))
    rows, bank = [], []
    for i, (r, m) in enumerate(zip(original, manifest), 1):
        if r['text'] != m['original_input_text'] or r['true_label'] != m['true_label']:
            raise ValueError(f'Baseline manifest differs at row {i}')
        sid = f'PAPER3000-{i:04}'
        text_hash = hashlib.sha256(r['text'].encode()).hexdigest()
        rows.append(dict(input_id=sid+':clean', sample_id=sid, kind='clean',
                         true_label='SPAM' if r['true_label']=='1' else 'NOT_SPAM',
                         email_text=r['text'], original_row=i, original_text_sha256=text_hash,
                         ceas_candidate_ids=m['candidate_ceas_ids']))
        candidates = [pair for ceas in m['candidate_ceas_ids'].split('|') for pair in by_source.get(ceas, [])]
        if r['true_label']=='1' and candidates:
            h, match = min(candidates, key=lambda p: (p[0]['source_csv'], int(p[0]['source_result_row_1based'])))
            bank.append(dict(sample_id=sid, original_text_sha256=text_hash,
                             rewrite_text=h['first_rewrite_text'], source_csv=h['source_csv'],
                             source_result_row=h['source_result_row_1based'], historical_defender=h['source_model'],
                             match_type=match['match_type'], generation_contract='historical_unverified',
                             semantic_validity='NOT_ASSESSED'))
    assert sum(r['true_label']=='SPAM' for r in rows)==1500
    baseline_path=ROOT/'dataset/paper_baseline3000.csv'
    bank_path=ROOT/'dataset/paper_step1_bank.csv'
    write(baseline_path, rows); write(bank_path, bank)
    provenance=dict(source=str(source.relative_to(ROOT)), source_sha256=expected,
                    baseline_sha256=sha(baseline_path), bank_sha256=sha(bank_path),
                    baseline_rows=3000, spam_rows=1500, ham_rows=1500,
                    historical_step1_candidates=len(bank),
                    selection='lexicographic source CSV, then first source record; no outcome filtering',
                    limitations='Recovered GPT-OSS cohort, not proof of identical cohorts for all historical models. '
                    'Preserves original input text/order/duplicates. Bank reuse is a declared hybrid experiment; '
                    'historical prompt/checkpoint and semantic validity are not independently established.',
                    lineage_inputs={str(p.relative_to(ROOT)):sha(p) for p in [
                        audit/'baseline_3000_manifest.csv',ROOT/'historical/all_eight_models_recorded_results.csv',
                        ROOT/'analysis/ceas_match/historical_matches.csv']})
    (ROOT/'dataset/paper_baseline3000.provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(f'Prepared {len(rows)} baseline inputs and {len(bank)} historical Step-1 candidates')


if __name__=='__main__':
    main()
