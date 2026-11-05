"""Paper/source metric definitions; no classifier or attacker requests."""
from collections import Counter
import math
import re
import statistics

VERSION = 1
METRICS = ('sbert', 'bleu', 'rouge_l', 'jaccard', 'length_ratio', 'keyword_retention')
# Matches content_retention_analysis_all_models.py in the supplied research archive.
STOPWORDS = set('the a an is are was were be been being and or but if in on at to for of with by from that this it as your'.split())


def divide(a, b):
    return a / b if b else None


def percent(a, b):
    value = divide(a, b)
    return 100 * value if value is not None else None


def classification_metrics(counts):
    tp,tn,fp,fn=(counts[k] for k in ('TP','TN','FP','FN'))
    precision=[divide(tp,tp+fp),divide(tn,tn+fn)]
    recall=[divide(tp,tp+fn),divide(tn,tn+fp)]
    f1=[divide(2*tp,2*tp+fp+fn),divide(2*tn,2*tn+fp+fn)]
    supports=[tp+fn,tn+fp]
    def macro(values):
        return sum(values)/2 if all(v is not None for v in values) else None
    def weighted(values):
        if not sum(supports) or any(v is None and n for v,n in zip(values,supports)):return None
        return sum(v*n for v,n in zip(values,supports) if n)/sum(supports)
    return dict(spam_precision=precision[0],spam_f1=f1[0],
        ham_precision=precision[1],ham_recall=recall[1],ham_f1=f1[1],
        macro_precision=macro(precision),macro_recall=macro(recall),
        weighted_precision=weighted(precision),weighted_recall=weighted(recall),weighted_f1=weighted(f1))


def attack_metrics(ledger, rows, counts, baseline_only):
    eligible = [r for r in rows if r['true_label'] == 'SPAM' and ledger.ok(r['sample_id'], 'baseline')
                and ledger.ok(r['sample_id'], 'baseline')['label'] == 'SPAM']
    total = len(eligible)
    generated = sum(bool(ledger.ok(r['sample_id'], 'rewrite1')) for r in eligible)
    first_valid = sum(bool(ledger.ok(r['sample_id'], 'classify1')) for r in eligible)
    remaining = [r for r in eligible if ledger.ok(r['sample_id'], 'classify1')
                 and ledger.ok(r['sample_id'], 'classify1')['label'] == 'SPAM']
    second_valid = sum(bool(ledger.ok(r['sample_id'], 'classify2')) for r in remaining)
    s1, s2 = counts['step1_flips'], counts['step2_additional_flips']
    unresolved = counts['attack_unresolved']
    complete = not baseline_only and not counts['baseline_errors'] and unresolved == 0
    first_complete = not baseline_only and not counts['baseline_errors'] and first_valid == total
    final = percent(s1 + s2, total) if complete else None
    effective = percent(s1 + s2, generated) if complete else None
    return dict(
        **classification_metrics(counts),
        paper_tp_pool=total, paper_step1_classified=first_valid,
        paper_step2_pool=len(remaining), paper_step2_classified=second_valid,
        paper_step1_asr_pct=percent(s1, total) if first_complete else None,
        paper_step2_gain_pct=percent(s2, len(remaining)) if complete else None,
        paper_final_asr_pct=final,
        paper_robust_accuracy_pct=100 - final if final is not None else None,
        paper_asr_lower_bound_pct=percent(s1+s2, total),
        paper_asr_upper_bound_pct=percent(s1+s2+unresolved, total),
        source_modified_rows=generated, source_unmodified_rows=total-generated,
        source_modified_asr_pct=effective,
        source_effective_robust_accuracy_pct=100-effective if effective is not None else None,
        source_observed_modified_flip_pct=percent(s1+s2, generated),
        paper_rates_complete=complete,
        paper_asr_definition='Label flips / true-positive spam pool; semantic validity separately assessed',
    )


def tokenize(text):
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def lexical_scores(original, rewrite):
    """Frozen, documented defaults where the paper does not specify settings."""
    ref, hyp = tokenize(original), tokenize(rewrite)
    if not ref or not hyp:
        return dict(bleu=None, rouge_l=None, jaccard=None,
                    length_ratio=len(hyp)/len(ref) if ref else None, keyword_retention=None)
    rset, hset = set(ref), set(hyp)
    # Sentence BLEU-4, uniform weights, Chen-Cherry method1 epsilon=.1.
    precisions = []
    for n in range(1, 5):
        rc = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        hc = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
        match = sum((rc & hc).values())
        precisions.append((match if match else .1) / max(1, sum(hc.values())))
    bp = 1 if len(hyp) >= len(ref) else math.exp(1-len(ref)/len(hyp))
    bleu = bp * math.exp(sum(math.log(p) for p in precisions)/4) if rset & hset else 0.0
    # Exact token LCS, using a bitset to avoid quadratic Python loops on emails.
    masks = {}
    for i, token in enumerate(ref):
        masks[token] = masks.get(token, 0) | (1 << i)
    state = 0
    for token in hyp:
        match = state | masks.get(token, 0)
        state = match & ~(match - ((state << 1) | 1))
    lcs = bin(state).count('1')
    keywords = {w for w, _ in Counter(w for w in ref if w not in STOPWORDS and len(w)>2).most_common(10)}
    return dict(bleu=bleu, rouge_l=2*lcs/(len(ref)+len(hyp)),
                jaccard=len(rset & hset)/len(rset | hset), length_ratio=len(hyp)/len(ref),
                keyword_retention=len(keywords & hset)/len(keywords) if keywords else None)


def semantic_band(value):
    if value is None:
        return 'NOT_COMPUTED'
    return 'strong' if value > .80 else 'moderate' if value >= .75 else 'potential_drift'


def similarity_summary(rows, ttest=None, stages=('step1', 'step2', 'final')):
    """Outcome-specific means/SD, delta(success-failure), independent t-tests."""
    output = []
    for stage in stages:
        selected = [r for r in rows if r['stage'] == stage]
        for metric in METRICS:
            groups = {g: [r[metric] for r in selected if (g == 'all' or r['outcome'] == g)
                          and isinstance(r.get(metric), (float, int)) and math.isfinite(r[metric])]
                      for g in ('all', 'success', 'failure', 'unresolved')}
            s, f = groups['success'], groups['failure']
            delta = statistics.mean(s)-statistics.mean(f) if s and f else None
            test = dict(t_statistic=None, p_value=None, test_status='not_requested', test_method='')
            if metric == 'sbert':
                test['test_method'] = 'Welch independent two-sided t-test; unadjusted descriptive p-value'
                if len(s)<2 or len(f)<2:
                    test['test_status'] = 'insufficient_group_size'
                elif ttest is not None:
                    if statistics.variance(s)==0 and statistics.variance(f)==0:
                        test['test_status'] = 'degenerate_variance'
                    else:
                        result = ttest(s, f, equal_var=False)
                        if math.isfinite(float(result.statistic)) and math.isfinite(float(result.pvalue)):
                            test.update(t_statistic=float(result.statistic), p_value=float(result.pvalue), test_status='OK')
                        else:
                            test['test_status'] = 'nonfinite_result'
                else:
                    test['test_status'] = 'dependency_or_sbert_unavailable'
            for group, values in groups.items():
                output.append(dict(stage=stage, metric=metric, outcome=group, n=len(values),
                    mean=statistics.mean(values) if values else None,
                    std=statistics.stdev(values) if len(values)>1 else None,
                    delta_success_minus_failure=delta, **test))
    return output
