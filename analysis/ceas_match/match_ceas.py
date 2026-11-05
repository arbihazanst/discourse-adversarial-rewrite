"""Read-only lineage audit against a separately downloaded CEAS corpus.

Run: python analysis/ceas_match/match_ceas.py
Writes audit outputs beside this script; never modifies benchmark/historical data.
Fuzzy candidates are suggestions, not automatically accepted identity matches.
"""
import collections
import csv
import difflib
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
csv.field_size_limit(100_000_000)


def read_csv(path, delimiter=','):
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig', errors='surrogateescape')
    text = ''.join(bytes([ord(c)-0xDC00]).decode('cp1252', errors='replace')
                   if 0xDC80 <= ord(c) <= 0xDCFF else c for c in text)
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def write_csv(name, rows, fields):
    with (HERE/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def ws(text):
    return ' '.join(text.casefold().split())


def typography(text):
    return ws(unicodedata.normalize('NFKC', text).translate(str.maketrans({
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...', '\ufeff': ''})))


def body(text):
    # Only remove a subject line where the recorded input actually has one.
    parts = text.replace('\r\n', '\n').replace('\r', '\n').split('\n', 1)
    return parts[1] if len(parts) == 2 and parts[0].lower().startswith('subject:') else text


def similarity(left, right):
    a, b = typography(left), typography(right)
    ta, tb = set(re.findall(r'\w+', a)), set(re.findall(r'\w+', b))
    return {
        'normalized_sequence_similarity': round(1.0 if a == b else
            difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 6),
        'token_jaccard': round(len(ta & tb)/len(ta | tb), 6) if ta | tb else 1.0,
        'recorded_to_source_length_ratio': round(len(left)/len(right), 6) if right else '',
    }


def main():
    source_path = HERE/'CEAS_08.csv'
    source = read_csv(source_path)
    texts = ['Subject: '+r['subject']+'\n\n'+r['body'] for r in source]
    normalizers = [('exact', lambda t: t),
                   ('line_endings', lambda t: t.replace('\r\n', '\n').replace('\r', '\n')),
                   ('whitespace_case', ws), ('typography', typography)]
    indexes = {}
    for name, norm in normalizers:
        idx = collections.defaultdict(list)
        for i, text in enumerate(texts, 1):
            idx[digest(norm(text))].append(i)
        indexes[name] = idx
    body_index = collections.defaultdict(list)
    for i, row in enumerate(source, 1):
        if ws(row['body']):
            body_index[digest(ws(row['body']))].append(i)
    old_rows = read_csv(ROOT/'lineage/CEAS_attack_results_full_crosswalk.csv')
    old = {(r['attack_file'], int(r['attack_record_number_1_based'])): r for r in old_rows}
    files = sorted((ROOT/'historical').glob('raw_*/*.csv'))
    records = []
    inputs = {}
    for path in files:
        rows = read_csv(path, ';')
        method = 'LLM' if path.name.startswith('LLM_') else 'GAN'
        field = 'originaltext' if method == 'LLM' else 'text'
        if rows and field not in rows[0]:
            raise ValueError(f'Missing original-text column in {path}')
        for n, row in enumerate(rows, 1):
            text = row[field]
            key = digest(text)
            inputs[key] = text
            records.append({'attack_file': path.name, 'method': method,
                            'attack_record_number_1_based': n,
                            'original_text_sha256': key})
    matches = {}
    for key, text in inputs.items():
        kind, candidates = 'unresolved', []
        if text.strip():
            for name, norm in normalizers:
                candidates = indexes[name].get(digest(norm(text)), [])
                if candidates:
                    kind = name
                    break
            if not candidates:
                candidates = body_index.get(digest(ws(body(text))), [])
                if candidates:
                    kind = 'body_only_whitespace_case'
        matches[key] = {'match_type': kind, 'candidates': candidates}
    unresolved = [k for k, m in matches.items() if not m['candidates']]
    print(f'Source={len(source)}, historical rows={len(records)}, unique raw texts={len(inputs)}, unresolved texts={len(unresolved)}', flush=True)
    fuzzy = []
    if unresolved:
        # Retrieve candidates from the full corpus, without restricting to spam or old IDs.
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(lowercase=True, strip_accents='unicode',
                                    sublinear_tf=True, max_features=150_000, dtype=np.float32)
        matrix = vectorizer.fit_transform(texts)
        query_matrix = vectorizer.transform([inputs[k] for k in unresolved])
        for key, query in zip(unresolved, query_matrix):
            scores = (query @ matrix.T).toarray().ravel()
            # Cosine often ties when only punctuation/encoding differs. Rerank
            # the top 20 plus ALL best-score ties by character similarity.
            ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
            pool = set(ranked[:20]) | set(np.flatnonzero(scores >= float(scores.max())-1e-6))
            similarities = {i: similarity(inputs[key], texts[i]) for i in pool}
            top = sorted(pool, key=lambda i: (-round(float(scores[i]), 5),
                -similarities[i]['normalized_sequence_similarity'], i))[:5]
            for rank, pos in enumerate(top, 1):
                fuzzy.append({'original_text_sha256': key, 'rank': rank,
                    'candidate_ceas_record_id_1based': pos+1, 'source_label': source[pos]['label'],
                    'tfidf_cosine': round(float(scores[pos]), 6),
                    **similarities[pos],
                    'recorded_original_text': inputs[key], 'candidate_source_text': texts[pos],
                    'identity_status': 'REVIEW_REQUIRED'})
    fuzzy_best = {r['original_text_sha256']: r for r in fuzzy if r['rank'] == 1}
    results, old_checks, counts = [], [], collections.Counter()
    usage = collections.defaultdict(lambda: {'candidate_rows': 0, 'unambiguous_rows': 0, 'files': set()})
    old_sim = {}
    for record in records:
        key = record['original_text_sha256']
        match = matches[key]
        ids = match['candidates']
        previous = old.get((record['attack_file'], record['attack_record_number_1_based']))
        previous_id = int(previous['CEAS_record_number_1_based']) if previous else None
        if previous_id is not None and not 1 <= previous_id <= len(source):
            raise ValueError('Old ID out of range')
        best = fuzzy_best.get(key, {})
        result = {**record, 'match_type': match['match_type'],
                  'candidate_count': len(ids), 'candidate_ceas_ids': '|'.join(map(str, ids)),
                  'unique_ceas_id': ids[0] if len(ids) == 1 else '',
                  'candidate_source_labels': '|'.join(sorted({source[i-1]['label'] for i in ids})),
                  'old_ceas_id': previous_id or '',
                  'old_id_in_candidates': int(previous_id in ids) if previous else '',
                  'old_match_type': previous['match_type'] if previous else '',
                  'fuzzy_best_candidate_id': best.get('candidate_ceas_record_id_1based', ''),
                  'fuzzy_best_cosine': best.get('tfidf_cosine', ''),
                  'fuzzy_best_sequence_similarity': best.get('normalized_sequence_similarity', ''),
                  'identity_status': ('REVIEW_REQUIRED' if not ids or match['match_type'].startswith('body_only')
                                      else 'AMBIGUOUS_DUPLICATES' if len(ids)>1 else 'MATCHED')}
        results.append(result)
        counts[match['match_type']] += 1
        for i in ids:
            usage[i]['candidate_rows'] += 1
            usage[i]['unambiguous_rows'] += int(len(ids) == 1)
            usage[i]['files'].add(record['attack_file'])
        if previous_id:
            sk = (key, previous_id)
            if sk not in old_sim:
                old_sim[sk] = similarity(inputs[key], texts[previous_id-1])
            old_checks.append({**record, 'old_ceas_id': previous_id,
                'old_match_type': previous['match_type'], 'new_match_type': match['match_type'],
                'old_id_in_new_candidates': int(previous_id in ids),
                'source_label': source[previous_id-1]['label'], **old_sim[sk]})
    clean = read_csv(ROOT/'dataset/clean_binary_balanced1484.csv')
    clean_results = []
    for row in clean:
        sid = int(row['ceas_record_id_1based'])
        clean_results.append({'sample_id': row['sample_id'], 'ceas_id': sid,
            'exact_email_match': int(row['email_text'] == texts[sid-1]),
            'subject_match': int(row['subject'] == source[sid-1]['subject']),
            'body_match': int(row['body'] == source[sid-1]['body']),
            'label_match': int(row['label'] == source[sid-1]['label'])})
    primary = read_csv(ROOT/'dataset/frozen_primary_step1_one_per_spam742.csv')
    historic = read_csv(ROOT/'historical/all_eight_models_recorded_results.csv')
    hist_index = {(r['source_csv'], int(r['source_result_row_1based'])): r for r in historic}
    rewrite_checks = []
    for row in primary:
        hist = hist_index[(row['source_csv'], int(row['historical_source_result_row_1based']))]
        sid = int(row['ceas_record_id_1based'])
        rewrite_checks.append({'sample_id': row['sample_id'], 'ceas_id': sid,
            'original_exact_match_download': int(row['original_text'] == texts[sid-1]),
            'rewrite_exact_match_historical': int(row['rewrite_text'] == hist['first_rewrite_text']),
            'source_id_matches_historical': int(sid == int(hist['ceas_record_id_1based']))})
    per_file = []
    for filename in sorted({r['attack_file'] for r in results}):
        subset = [r for r in results if r['attack_file'] == filename]
        per_file.append({'attack_file': filename, 'rows': len(subset),
            **{k: sum(r['match_type'] == k for r in subset) for k in counts},
            'ambiguous_rows': sum(r['candidate_count']>1 for r in subset),
            'review_required_rows': sum(r['identity_status']=='REVIEW_REQUIRED' for r in subset),
            'old_id_in_candidates': sum(r['old_id_in_candidates']==1 for r in subset)})
    outputs = [('historical_matches.csv', results), ('previous_mapping_verification.csv', old_checks),
               ('match_summary_by_file.csv', per_file), ('benchmark_clean_verification.csv', clean_results),
               ('primary_rewrite_lineage_verification.csv', rewrite_checks)]
    for name, rows in outputs:
        write_csv(name, rows, list(rows[0]))
    write_csv('fuzzy_candidates_review.csv', fuzzy,
              list(fuzzy[0]) if fuzzy else ['original_text_sha256', 'identity_status'])
    review_groups = {}
    for record in records:
        key = record['original_text_sha256']
        if matches[key]['match_type'] != 'unresolved':
            continue
        prior = old.get((record['attack_file'], record['attack_record_number_1_based']))
        if not prior:
            continue
        sid = int(prior['CEAS_record_number_1_based'])
        sk = (key, sid)
        if sk not in review_groups:
            left, right = typography(inputs[key]), typography(texts[sid-1])
            edits = [{'operation': tag, 'recorded': left[a:b], 'downloaded': right[c:d]}
                for tag,a,b,c,d in difflib.SequenceMatcher(None,left,right,autojunk=False).get_opcodes()
                if tag != 'equal']
            review_groups[sk] = {'original_text_sha256': key, 'old_ceas_id': sid,
                'old_match_type': prior['match_type'], 'historical_rows': 0,
                **old_sim[sk], 'normalized_text_differences_json': json.dumps(edits, ensure_ascii=False),
                'recorded_original_text': inputs[key], 'downloaded_source_text': texts[sid-1],
                'identity_status': 'REVIEW_REQUIRED'}
        review_groups[sk]['historical_rows'] += 1
    reviews = list(review_groups.values())
    write_csv('encoding_differences_review.csv', reviews,
              list(reviews[0]) if reviews else ['original_text_sha256','identity_status'])
    usage_rows = [{'ceas_record_id_1based': i, 'label': row['label'],
        'source_email_sha256': digest(texts[i-1]), 'historical_candidate_rows': usage[i]['candidate_rows'],
        'historical_unambiguous_rows': usage[i]['unambiguous_rows'],
        'historical_files': '|'.join(sorted(usage[i]['files']))} for i,row in enumerate(source,1)]
    write_csv('ceas_source_usage.csv', usage_rows, list(usage_rows[0]))
    summary = {
        'source_rows': len(source), 'source_labels': dict(collections.Counter(r['label'] for r in source)),
        'historical_rows': len(records), 'historical_files': len(files), 'unique_recorded_raw_texts': len(inputs),
        'matching_counts': dict(counts), 'ambiguous_candidate_rows': sum(r['candidate_count']>1 for r in results),
        'old_mapping_ids_recovered_automatically': sum(r['old_id_in_candidates']==1 for r in results),
        'old_mapping_ids_not_recovered_automatically': sum(r['old_id_in_candidates']==0 for r in results),
        'unresolved_unique_texts': len(unresolved),
        'fuzzy_best_agrees_with_old_id_rows': sum(r['fuzzy_best_candidate_id']==r['old_ceas_id'] for r in results if r['match_type']=='unresolved'),
        'clean_benchmark_rows': len(clean_results),
        'clean_benchmark_all_fields_exact': sum(all(r[k] for k in ['exact_email_match','subject_match','body_match','label_match']) for r in clean_results),
        'primary_rewrite_rows': len(rewrite_checks),
        'primary_rewrite_all_checks_pass': sum(all(r[k] for k in ['original_exact_match_download','rewrite_exact_match_historical','source_id_matches_historical']) for r in rewrite_checks),
        'historical_llm_distinct_normalized_originals': len({ws(r['original_text_as_recorded']) for r in historic}),
        'historical_llm_distinct_mapped_source_ids': len({r['ceas_record_id_1based'] for r in historic}),
        'original_3000_baseline_recovered': False,
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files+[source_path, ROOT/'lineage/CEAS_attack_results_full_crosswalk.csv',
                             ROOT/'dataset/clean_binary_balanced1484.csv', ROOT/'dataset/frozen_primary_step1_one_per_spam742.csv',
                             ROOT/'historical/all_eight_models_recorded_results.csv']},
    }
    (HERE/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
