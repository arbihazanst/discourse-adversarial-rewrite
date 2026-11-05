#!/usr/bin/env python3
"""Compute clean classification metrics and paired original->rewrite label flips.

Python 3.9+, standard library. Missing/incomplete predictions never become NOT_SPAM.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path


def safe_div(a, b):
    return a / b if b else None


def percent(value):
    return f"{100*value:.2f}%" if value is not None else "NA"


IDENTITY = ("run_id", "defender", "backend_mode", "model_id", "checkpoint_id", "dataset_sha256", "endpoint", "tool_id")


def load(path, dataset_hash=None):
    d = {}
    identities = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not all(r.get(k) for k in ("run_id", "defender", "input_id", "backend_mode", "model_id", "dataset_sha256", "endpoint")):
                raise SystemExit("Missing run identity in predictions; legacy results need a new benchmark run.")
            if dataset_hash and r["dataset_sha256"] != dataset_hash:
                raise SystemExit("Predictions were produced using a different dataset.")
            identity = tuple(r.get(k, "") for k in IDENTITY)
            if r["run_id"] in identities and identities[r["run_id"]] != identity:
                raise SystemExit("Conflicting metadata for the same run_id.")
            identities[r["run_id"]] = identity
            d[(identity, r["input_id"])] = r  # latest attempt within this run only
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/decision_metrics.csv"))
    args = ap.parse_args()
    with args.dataset.open(encoding="utf-8-sig", newline="") as f:
        samples = list(csv.DictReader(f))
    pred = load(args.predictions, hashlib.sha256(args.dataset.read_bytes()).hexdigest())
    defenders = sorted({k[0] for k in pred})
    if not defenders:
        raise SystemExit("No prediction records found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(IDENTITY) + ["clean_expected", "clean_valid", "clean_missing_or_error",
            "clean_accuracy", "clean_macro_f1", "clean_spam_recall", "clean_fpr",
            "attack_expected", "attack_valid", "attack_missing_or_error",
            "initially_spam_with_valid_attack", "observed_flips", "label_flip_rate",
            "initially_spam_attack_unavailable", "initially_not_spam", "matched_pairs"]
    outrows = []
    for model in defenders:
        count_fields = ("clean_valid", "clean_missing_or_error", "attack_valid", "attack_missing_or_error",
                        "initially_spam_with_valid_attack", "observed_flips", "initially_spam_attack_unavailable",
                        "initially_not_spam", "matched_pairs")
        stats = defaultdict(int, dict.fromkeys(count_fields, 0))
        originals, attacks = {}, {}
        for row in samples:
            if row["kind"] == "clean":
                originals[row["sample_id"]] = row if row["true_label"] == "SPAM" else originals.get(row["sample_id"], row)
            elif row["kind"] == "frozen_primary_step1":
                attacks[row["sample_id"]] = row
        tp=tn=fp=fn=0
        clean_records = [r for r in samples if r["kind"] == "clean"]
        attack_records = [r for r in samples if r["kind"] == "frozen_primary_step1"]
        for r in clean_records:
            p = pred.get((model,r["input_id"]))
            l = p.get("predicted_label") if p and p.get("status") == "OK" else None
            if l not in ("SPAM","NOT_SPAM"):
                stats["clean_missing_or_error"]+=1
                continue
            stats["clean_valid"]+=1
            t=r["true_label"]
            if (t,l)==("SPAM","SPAM"):tp+=1
            if (t,l)==("SPAM","NOT_SPAM"):fn+=1
            if (t,l)==("NOT_SPAM","SPAM"):fp+=1
            if (t,l)==("NOT_SPAM","NOT_SPAM"):tn+=1
        for r in attack_records:
            p = pred.get((model,r["input_id"]))
            l = p.get("predicted_label") if p and p.get("status") == "OK" else None
            if l not in ("SPAM","NOT_SPAM"):
                stats["attack_missing_or_error"]+=1
            else:
                stats["attack_valid"]+=1
        for sid,a in attacks.items():
            orig=originals.get(sid)
            if not orig or orig["true_label"]!="SPAM":
                continue
            p0=pred.get((model,orig["input_id"]))
            p1=pred.get((model,a["input_id"]))
            y0=p0.get("predicted_label") if p0 and p0.get("status")=="OK" else None
            y1=p1.get("predicted_label") if p1 and p1.get("status")=="OK" else None
            if y0 not in ("SPAM","NOT_SPAM"):
                continue
            if y0=="NOT_SPAM":
                stats["initially_not_spam"]+=1
                continue
            if y1 not in ("SPAM","NOT_SPAM"):
                stats["initially_spam_attack_unavailable"]+=1
                continue
            stats["initially_spam_with_valid_attack"]+=1
            stats["matched_pairs"]+=1
            if y1=="NOT_SPAM":
                stats["observed_flips"]+=1
        pprec=safe_div(tp,tp+fp); prec_not=safe_div(tn,tn+fn)
        recall_spam=safe_div(tp,tp+fn); recall_not=safe_div(tn,tn+fp)
        f1_spam=safe_div(2*tp,2*tp+fp+fn)
        f1_not=safe_div(2*tn,2*tn+fp+fn)
        mf1=(f1_spam+f1_not)/2 if f1_spam is not None and f1_not is not None else None
        output = dict(stats)
        output.update({**dict(zip(IDENTITY, model)), "clean_expected":len(clean_records),"attack_expected":len(attack_records),
            "clean_accuracy":percent(safe_div(tp+tn,tp+tn+fp+fn)),
            "clean_macro_f1":percent(mf1), "clean_spam_recall":percent(recall_spam),
            "clean_fpr":percent(safe_div(fp,fp+tn)),
            "label_flip_rate":percent(safe_div(stats["observed_flips"],stats["initially_spam_with_valid_attack"]))})
        outrows.append(output)
    with args.out.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader();w.writerows(outrows)
    for r in outrows:
        print(f"{r['defender']} [{r['backend_mode']} {r['run_id'][:12]}]: clean={r['clean_valid']}/{r['clean_expected']} accuracy={r['clean_accuracy']} macro-F1={r['clean_macro_f1']} flip={r['observed_flips']}/{r['initially_spam_with_valid_attack']} ({r['label_flip_rate']})")
    print(f"Results: {args.out}")

if __name__=="__main__":main()
