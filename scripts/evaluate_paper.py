"""Evaluate saved adaptive runs without contacting classifiers or regenerating text."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time

import adaptive_pipeline as pipeline
import paper_evaluation as metrics

DEFAULT_SBERT = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
PAIR_FIELDS = ['defender','sample_id','stage','source_step','outcome','predicted_label',
               'original_sha256','rewrite_sha256',*metrics.METRICS,'semantic_band',
               'original_tokens','rewrite_tokens','sbert_original_truncated','sbert_rewrite_truncated',
               'sbert_status','source','cache_key']


class SbertScorer:
    def __init__(self, args):
        try:
            from sentence_transformers import SentenceTransformer
            from scipy.stats import ttest_ind
        except ImportError as exc:
            raise ValueError('Full evaluation needs scripts/requirements-evaluation.txt; use --lexical-only to skip SBERT') from exc
        if not re.fullmatch(r'[0-9a-f]{40}', args.sbert_revision):
            raise ValueError('--sbert-revision must be an immutable 40-character Hugging Face commit SHA')
        self.model = SentenceTransformer(args.sbert_model, revision=args.sbert_revision, device=args.device,
                                         cache_folder=str(args.model_cache), local_files_only=args.local_files_only,
                                         trust_remote_code=False)
        self.ttest = ttest_ind
        self.contract = dict(model=args.sbert_model, revision=args.sbert_revision,
            max_seq_length=self.model.max_seq_length, device=args.device,
            truncation='model default; flags recorded for both texts',
            packages={name:importlib.metadata.version(name) for name in
                      ('sentence-transformers','transformers','torch','numpy','scipy')})

    def score(self, original, rewrite):
        texts=[original,rewrite]
        embeddings=self.model.encode(texts,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        value=float(embeddings[0] @ embeddings[1])
        if not math.isfinite(value):raise ValueError('Non-finite SBERT score')
        lengths=[len(self.model.tokenizer(t, truncation=False, add_special_tokens=True)['input_ids']) for t in texts]
        return dict(sbert=max(-1.0,min(1.0,value)),sbert_status='OK',
                    sbert_original_truncated=lengths[0]>self.model.max_seq_length,
                    sbert_rewrite_truncated=lengths[1]>self.model.max_seq_length)


def load_run(root, model, dataset_override=None):
    pipeline.validate_model_key(model)
    folder=root/model
    saved=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    dataset=pipeline.rooted(dataset_override or saved['dataset'])
    rows=pipeline.api.load_dataset(dataset,{'clean'})
    by_id={r['sample_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate sample IDs')
    baseline=pipeline.fingerprint(dict(dataset=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        defender=saved['defender'],questions=pipeline.api.SYSTEMONE_QUESTION,version=1))
    attack=pipeline.fingerprint({'baseline':baseline,**saved['attack_contract']})
    if saved['identities']!={'baseline':baseline,'attack':attack}:
        raise ValueError('Saved run/dataset identity mismatch')
    selected=saved.get('selected_sample_ids')
    if selected is None:
        checkpoint=folder/'checkpoint.json'
        try:selected=json.loads(checkpoint.read_text(encoding='utf-8')).get('selected_sample_ids')
        except (OSError,ValueError):pass
    if selected is None:
        # Old runs without a selection manifest: evaluate full declared dataset,
        # explicitly retaining unclassified inputs as missing.
        selected=list(by_id)
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=by_id.keys():
        raise ValueError('Invalid saved selection')
    if saved.get('attack_kind')=='gan':
        from gan_pipeline import GanLedger
        ledger=GanLedger(root,model,saved['identities'],set(by_id))
    else:ledger=pipeline.Ledger(root,model,saved['identities'],set(by_id))
    return ledger,[by_id[sid] for sid in selected],saved


def evaluate_run(ledger, rows, saved, scorer=None, lexical_only=True):
    folder=ledger.folder/'evaluation'
    is_gan=saved.get('attack_kind')=='gan'
    contract=dict(evaluation_version=metrics.VERSION,
        lexical='lowercase ASCII alphanumeric tokens; BLEU4 uniform method1 epsilon=.1; ROUGE-L F1 no stemming',
        content_retention='top10 frequency tokens length>2; supplied archive stopwords; empty keyword set missing',
        sbert=scorer.contract if scorer else 'NOT_COMPUTED_LEXICAL_ONLY')
    identity=pipeline.fingerprint(contract)
    cache=ledger.root/'evaluation_cache'/identity
    pairs=[]; start=time.perf_counter()
    print(f'{ledger.folder.name}: evaluating saved rewrites ({"lexical only" if lexical_only else "full"})',flush=True)
    for row in rows:
        sid=row['sample_id'];base=ledger.ok(sid,'baseline')
        if row['true_label']!='SPAM' or not base or base['label']!='SPAM':continue
        stages={};specs=[]
        if is_gan:
            probe=ledger.get(sid,'gan_probe')
            if probe and probe.get('artifact'):
                specs.append(('gan_candidate',1,ledger.artifact(probe)['text'],ledger.ok(sid,'gan_probe'),'openattack_gan_sst_probe'))
            attack=ledger.ok(sid,'gan_attack')
            result=ledger.artifact(attack) if attack else {}
            if result.get('returned_text'):
                specs.append(('gan_returned',1,result['returned_text'],ledger.ok(sid,'gan_classify'),'openattack_gan_sst_returned'))
        else:
            for step in (1,2):
                gen=ledger.ok(sid,'rewrite'+str(step));first=ledger.ok(sid,'classify1')
                if not gen or (step==2 and (not first or first['label']!='SPAM')):continue
                specs.append(('step'+str(step),step,ledger.text(sid,'rewrite'+str(step)),ledger.ok(sid,'classify'+str(step)),gen['source']))
        for stage,step,rewrite,decision,source in specs:
            original=row['email_text']
            oh=hashlib.sha256(original.encode()).hexdigest();rh=hashlib.sha256(rewrite.encode()).hexdigest()
            key=pipeline.fingerprint(dict(original=oh,rewrite=rh,contract=contract))
            cached=cache/(key+'.json')
            with pipeline.file_lock(cache/(key+'.lock')):
                if cached.exists():
                    envelope=json.loads(cached.read_text(encoding='utf-8'))
                    values=envelope['values']
                    if envelope['key']!=key or envelope['sha256']!=pipeline.fingerprint(values):
                        raise ValueError('Evaluation cache checksum mismatch')
                else:
                    values=metrics.lexical_scores(original,rewrite)
                    values.update(original_tokens=len(metrics.tokenize(original)),rewrite_tokens=len(metrics.tokenize(rewrite)))
                    values.update(scorer.score(original,rewrite) if scorer and rewrite.strip() else
                        dict(sbert=None,sbert_status='EMPTY_TEXT' if not rewrite.strip() else 'NOT_COMPUTED',sbert_original_truncated=None,sbert_rewrite_truncated=None))
                    pipeline.atomic_json(cached,dict(key=key,values=values,sha256=pipeline.fingerprint(values)))
            outcome=('success' if decision['label']=='NOT_SPAM' else 'failure') if decision else 'unresolved'
            pair=dict(defender=ledger.folder.name,sample_id=sid,stage=stage,source_step=step,
                outcome=outcome,predicted_label=decision['label'] if decision else '',original_sha256=oh,rewrite_sha256=rh,
                **values,semantic_band=metrics.semantic_band(values['sbert']),source=source,cache_key=key)
            pairs.append(pair); stages[step]=pair
            if len(pairs)%50==0:print(f'  scored {len(pairs)} rows | elapsed={time.perf_counter()-start:.1f}s',flush=True)
        # Final candidate is Step1 success or a classified Step2. Never fall
        # back to Step1 failure while Step2 is missing/unresolved.
        final=stages.get(1) if stages.get(1,{}).get('outcome')=='success' else stages.get(2)
        if not is_gan and final and final['outcome']!='unresolved':pairs.append({**final,'stage':'final'})
    summaries=metrics.similarity_summary(pairs,scorer.ttest if scorer else None,
        stages=('gan_candidate','gan_returned') if is_gan else ('step1','step2','final'))
    pipeline.atomic_csv(folder/'similarity_per_rewrite.csv',pairs,PAIR_FIELDS)
    pipeline.atomic_csv(folder/'similarity_summary.csv',summaries,list(summaries[0]))
    # Rebuild base/attack metrics and CSV exports from the validated ledger only.
    baseline_only=saved.get('baseline_only',not any(stage!='baseline' for _,stage in ledger.events))
    if is_gan:
        from gan_pipeline import report
        report(ledger,rows,baseline_only)
    else:pipeline.report(ledger,rows,baseline_only)
    current=json.loads((ledger.folder/'metrics.json').read_text(encoding='utf-8'))
    summary=dict(contract=contract,evaluation_id=identity,model=ledger.folder.name,selected_rows=len(rows),attack_kind='gan' if is_gan else 'llm',
        source_events_sha256=hashlib.sha256(ledger.path.read_bytes()).hexdigest(),
        scored_step1=sum(r['stage']=='step1' for r in pairs),scored_step2=sum(r['stage']=='step2' for r in pairs),
        final_candidates=sum(r['stage']=='final' for r in pairs),
        gan_candidates=sum(r['stage']=='gan_candidate' for r in pairs),gan_returned=sum(r['stage']=='gan_returned' for r in pairs),
        sbert_status='NOT_COMPUTED' if lexical_only else 'COMPUTED',
        semantic_validity='NOT_ASSESSED_BY_HUMAN',
        statistical_limitations=('Descriptive unadjusted tests; preserved duplicates violate strict independence; '
                                +('GAN returned candidates are selected by probe success' if is_gan else 'Step2 is conditional')),
        baseline_and_attack_metrics=current)
    pipeline.atomic_json(folder/'evaluation.json',summary)
    pipeline.atomic_csv(folder/'paper_metrics.csv',[{k:v for k,v in current.items() if k!='identities'}],
                        [k for k in current if k!='identities'])
    pipeline.atomic_json(folder/'checkpoint.json',dict(state='complete',evaluation_id=identity,
                         source_events_sha256=summary['source_events_sha256'],scored_pairs=len(pairs)))
    print(f'{ledger.folder.name}: saved evaluation to {folder}',flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=pipeline.ROOT/'outputs/paper3000_adaptive')
    parser.add_argument('--model',default='all')
    parser.add_argument('--dataset',type=Path,help='Relocated identical dataset; hash must match saved run')
    parser.add_argument('--lexical-only',action='store_true',help='No SBERT/dependency downloads; SBERT fields stay missing')
    parser.add_argument('--sbert-model',default=DEFAULT_SBERT)
    parser.add_argument('--sbert-revision',default=DEFAULT_REVISION)
    parser.add_argument('--model-cache',type=Path,default=pipeline.ROOT/'.cache/evaluation_models')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--local-files-only',action='store_true')
    args=parser.parse_args(argv)
    root=pipeline.rooted(args.run_dir)
    models=sorted(p.parent.name for p in root.glob('*/run.json')) if args.model=='all' else [args.model]
    if not models:raise ValueError('No saved runs found; run a classification pilot first')
    # Validate files before loading/downloading the optional embedding model.
    for name in models:load_run(root,name,args.dataset)
    args.model_cache=pipeline.rooted(args.model_cache)
    scorer=None if args.lexical_only else SbertScorer(args)
    for name in models:
        with pipeline.file_lock(root/name/'run.lock'):
            ledger,rows,saved=load_run(root,name,args.dataset)
            evaluate_run(ledger,rows,saved,scorer,args.lexical_only)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:
        print('Evaluation interrupted. Pair cache is saved; rerun the same command to resume.');sys.exit(130)
    except (ValueError,OSError,KeyError,ImportError) as exc:
        print(f'Evaluation error: {exc}',file=sys.stderr);sys.exit(2)
