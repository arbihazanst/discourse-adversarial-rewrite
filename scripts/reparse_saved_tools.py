"""Recover parse failures from hash-verified saved tool responses, without HTTP."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import adaptive_pipeline as core
import evaluate_paper


def recover(root,model='pplx'):
    with core.file_lock(root/model/'run.lock'):
        ledger,rows,saved=evaluate_paper.load_run(root,model)
        defender=saved['defender']
        if defender.get('transport')!='tool':raise ValueError('Recovery is only for tool responses')
        selected=[r['sample_id'] for r in rows];recovered=0
        for (sid,stage),old in list(ledger.events.items()):
            if sid not in selected or old['status']!='ERROR' or not old.get('artifact'):continue
            if stage not in ('baseline','classify1','classify2','gan_probe','gan_classify'):continue
            artifact=ledger.artifact(old)
            if 'response' not in artifact or 'request' not in artifact:continue
            if core.fingerprint(artifact['request'])!=old['request_sha256']:
                raise ValueError('Saved request fingerprint mismatch')
            request=artifact['request']
            if request.get('model')!=defender['model'] or request.get('tool_ids')!=[defender['tool_id']]:
                raise ValueError('Saved request does not match the run tool/model')
            try:
                text=None
                if defender.get('tool_protocol')=='legacy_fixed_spam_v2':
                    parameters=json.loads(request['messages'][1]['content'])
                    text=parameters['state']
                    if parameters!=core.api.tool_parameters(text):continue
                label,p_spam,p_not_spam=core.parse_decision(defender,artifact['response'],text)
            except ValueError:continue
            new=copy.deepcopy(artifact)
            new['reparse_provenance']={'original_artifact':old['artifact'],'original_sha256':old['artifact_sha256'],
                'parser_sha256':hashlib.sha256(Path(core.api.__file__).read_bytes()).hexdigest(),
                'note':'Recovered from saved tool output; no inference. Missing original timing stays missing.'}
            event={**old,'status':'OK','label':label,'p_spam':p_spam,'p_not_spam':p_not_spam,
                   'error':'','source':'reparsed_saved_response','artifact':'','artifact_sha256':''}
            ledger.commit(event,new,selected);recovered+=1
        if saved.get('attack_kind')=='gan':
            from gan_pipeline import report
        else:report=core.report
        complete=report(ledger,rows,saved.get('baseline_only',False))
        ledger.checkpoint('complete' if complete else 'incomplete',selected)
        print(f'{root/model}: recovered {recovered} saved responses; no model calls.')
        return recovered


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--model',default='pplx')
    args=p.parse_args(argv);core.validate_model_key(args.model)
    recover(core.rooted(args.run_dir),args.model);return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,KeyError) as exc:print('Recovery error:',exc,file=sys.stderr);sys.exit(2)
