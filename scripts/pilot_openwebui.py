"""PPLX pilot: five spam + five ham, with optional paper LLM or SST GAN attack."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import adaptive_pipeline as pipeline
from env_config import load_project_env, PROJECT_ENV


def get_json(url,key,timeout):
    request=Request(url,headers={'Authorization':'Bearer '+key,'Accept':'application/json'})
    try:
        with urlopen(request,timeout=timeout) as response:return json.load(response)
    except HTTPError as exc:
        raise ValueError(f'Open WebUI metadata HTTP {exc.code} at {url}; check API-key permissions and endpoint') from exc
    except (URLError,TimeoutError) as exc:
        raise ValueError(f'Cannot reach Open WebUI metadata at {url}: {exc}') from exc


def catalog_rows(value,kind):
    if isinstance(value,dict):value=value.get('data',value.get(kind))
    if not isinstance(value,list):raise ValueError('Unexpected Open WebUI '+kind+' catalog')
    return [row for row in value if isinstance(row,dict) and isinstance(row.get('id'),str)]


def select_tool(rows,hint):
    exact=[row for row in rows if row['id']==hint]
    if len(exact)==1:return exact[0]
    normalize=lambda value:re.sub('[^a-z0-9]','',value.lower())
    wanted=normalize(hint)
    matches=[row for row in rows if wanted and (normalize(row['id'])==wanted
             or normalize(str(row.get('name','')))==wanted)]
    if len(matches)==1:return matches[0]
    available=', '.join(row['id'] for row in rows) or '(none)'
    raise ValueError(f'Tool ID {hint!r} is missing or ambiguous. Available IDs: {available}. Use --tool-id EXACT_ID.')


def resolve_route(defender,timeout,list_only=False):
    suffix='/api/chat/completions'
    if not defender['url'].endswith(suffix):raise ValueError('Tool endpoint must end with /api/chat/completions')
    base=defender['url'][:-len(suffix)]
    key=pipeline.credentials(defender)
    models=catalog_rows(get_json(base+'/api/models',key,timeout),'models')
    tools=catalog_rows(get_json(base+'/api/v1/tools/',key,timeout),'tools')
    if list_only:
        print('Models:')
        for row in models:print(' ',row['id'])
        print('Tools (ID | name):')
        for row in tools:print(' ',row['id'],'|',row.get('name',''))
        return
    matches=[row for row in models if row['id']==defender['model']]
    if len(matches)!=1:
        raise ValueError('Chat model not available: '+defender['model']+'. Use --list-tools to inspect exact IDs.')
    tool=select_tool(tools,defender['tool_id'])
    defender['tool_id']=tool['id']
    # Hash API metadata, not credentials or complete tool source. This is NOT a
    # weight revision: the pilot must stay distinct from a pinned benchmark.
    snapshot={'model':{k:matches[0][k] for k in ('id','digest','ollama','info') if k in matches[0]},
              'tool':{k:tool[k] for k in ('id','name','specs') if k in tool}}
    defender['pilot_metadata_sha256']=pipeline.fingerprint(snapshot)
    print(f'PPLX route verified in catalog: chat={defender["model"]}, tool={defender["tool_id"]}')
    return snapshot


def reuse_legacy_catalog_identity(defender,snapshot,folders):
    """Prove old metadata differs ONLY by the server-generated updated_at value.

    Earlier pilots hashed this volatile timestamp. Recover its exact preimage
    near the first event, or from a saved compatibility proof; never ignore a
    changed model, tool, protocol, endpoint, revision, or catalog payload.
    """
    for folder in folders:
        run=folder/'pplx/run.json'
        if not run.exists():continue
        saved=json.loads(run.read_text(encoding='utf-8'))['defender']
        old=saved.get('pilot_metadata_sha256')
        if {k:v for k,v in saved.items() if k!='pilot_metadata_sha256'}!={k:v for k,v in defender.items() if k!='pilot_metadata_sha256'}:continue
        if old==defender['pilot_metadata_sha256']:return
        proof_path=folder/'pplx/pilot_catalog_compatibility.json'
        timestamps=[]
        if proof_path.exists():
            proof=json.loads(proof_path.read_text(encoding='utf-8'))
            if isinstance(proof.get('legacy_updated_at'),int):timestamps.append(proof['legacy_updated_at'])
        events=pipeline.read_csv(folder/'pplx/events.csv')
        if events:
            artifact=(folder/events[0]['artifact']).resolve()
            if artifact.is_relative_to(folder.resolve()) and artifact.is_file():
                raw=json.loads(artifact.read_text(encoding='utf-8'))
                first=int(raw.get('response',{}).get('created',artifact.stat().st_mtime))
                timestamps.extend(range(first-600,first+6))
        for stamp in timestamps:
            previous=copy.deepcopy(snapshot);previous['tool']['updated_at']=stamp
            if pipeline.fingerprint(previous)==old:
                pipeline.atomic_json(proof_path,{'legacy_updated_at':stamp,'legacy_metadata_sha256':old,
                    'stable_metadata_sha256':defender['pilot_metadata_sha256'],
                    'note':'Exact old hash reconstructed; only volatile tool.updated_at excluded from catalog comparison.'})
                defender['pilot_metadata_sha256']=old
                print('Resume catalog verified: only server-generated tool.updated_at changed; saved classifications reused.')
                return


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=pipeline.ROOT/'decision_models/adaptive_main.json')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--attack',choices=('none','llm','gan'),default='none')
    parser.add_argument('--gan-resource-dir',type=Path)
    parser.add_argument('--base-url',help='Open WebUI base URL, without /api/chat/completions')
    parser.add_argument('--tool-id');parser.add_argument('--chat-model')
    parser.add_argument('--profile-routes',action='store_true',help='Use model/endpoint/revision from the selected JSON profile; load only its API key from .env (used by the menu)')
    parser.add_argument('--checkpoint-id',help='Known combined tool/backend and orchestration revision, if available')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--check-server',action='store_true')
    parser.add_argument('--list-tools',action='store_true',help='List accessible tool/model IDs; no classification')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--retry-errors',action='store_true')
    parser.add_argument('--pilot',action='store_true',help='Always enabled; this launcher cannot run the full dataset')
    parser.add_argument('--timeout',type=float,default=300);parser.add_argument('--attempts',type=int,default=1)
    parser.add_argument('--delay',type=float,default=0)
    args=parser.parse_args(argv)
    if args.attempts<1 or not math.isfinite(args.timeout) or args.timeout<=0 or not math.isfinite(args.delay) or args.delay<0:
        raise ValueError('Invalid retry/timeout/delay')
    if args.dry_run and (args.check_server or args.list_tools):raise ValueError('Offline dry-run cannot query the server')
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if config.get('profile') not in {'main','experimenting'}:raise ValueError('Choose a main or experimenting profile')
    if config.get('protocol')=='validity_gated_v1' and config['profile']!='experimenting':
        raise ValueError('PPLX pilot is experimenting-only. Main uses Clef/d3 via run_cli.')
    config=copy.deepcopy(config);defender=config['defenders']['pplx']
    if defender.get('transport')!='tool':raise ValueError('PPLX pilot requires transport=tool in this profile')
    # Preserve a terminal chat-model alias over a differently named .env alias.
    terminal_chat_model=os.environ.get('PPLX_MODEL_ID') or os.environ.get('PPLX_CHAT_MODEL')
    load_project_env(allowed_keys={defender.get('key_env','OWUI_API_KEY')} if args.profile_routes else None)
    env={} if args.profile_routes else os.environ
    defender['tool_id']=args.tool_id or env.get('PPLX_TOOL_ID') or defender.get('tool_id','server:pplxdeciderflash')
    defender['model']=args.chat_model or (terminal_chat_model if not args.profile_routes else None) or env.get('PPLX_MODEL_ID') or env.get('PPLX_CHAT_MODEL') or defender['model']
    if args.base_url:defender['url']=args.base_url.rstrip('/')+'/api/chat/completions'
    elif env.get('PPLX_URL'):defender['url']=env['PPLX_URL']
    defender['revision']=args.checkpoint_id or env.get('PPLX_CHECKPOINT_ID') or defender.get('revision','')
    if not defender['revision']:
        defender['revision']='UNPINNED_CONNECTIVITY_PILOT'
        defender['revision_status']='UNVERIFIED_WEIGHTS'
    defender['run_purpose']='connectivity_pilot_10'
    pipeline.validate_endpoint(defender,live=False)
    dataset=pipeline.rooted(config['dataset'])
    rows=pipeline.api.load_dataset(dataset,{'clean'})
    if any(sum(row['true_label']==label for row in rows)<5 for label in ('SPAM','NOT_SPAM')):
        raise ValueError('Pilot requires at least five spam and five ham originals')
    prefix='experimenting/' if config['profile']=='experimenting' else ''
    baseline_output=pipeline.rooted(config.get('tool_pilot_output_dir',prefix+'outputs/pplx_gemma_fixed_pilot10'))
    output=pipeline.rooted(args.output_dir or (baseline_output if args.attack=='none' else
        config.get('tool_'+args.attack+'_pilot_output_dir',prefix+'outputs/pplx_'+args.attack+'_pilot10')))
    if config['profile']=='experimenting' and not output.is_relative_to((pipeline.ROOT/'experimenting').resolve()):
        raise ValueError('Pilot output must stay inside experimenting')
    if output==pipeline.rooted(config['output_dir']):raise ValueError('Use a separate pilot output folder, not the full experiment output')
    if args.attack!='none' and output==baseline_output:raise ValueError('Attack pilot must use a different output from its baseline source')
    config.update(defenders={'pplx':defender},active_models=['pplx'],default_model='pplx',output_dir=str(output))
    config['experiment_purpose']='connectivity_pilot10' if args.attack=='none' else args.attack+'_attack_pilot10_unverified'
    if args.attack=='gan':
        config['gan_output_dir']=str(output)
        if args.gan_resource_dir:config.setdefault('gan',{})['resource_dir']=str(args.gan_resource_dir)
    if args.attack=='llm':
        attacker=config['attacker']
        if not attacker.get('revision'):
            attacker.update(revision='UNPINNED_ATTACK_PILOT',revision_status='UNVERIFIED_WEIGHTS')
    print(f'PPLX ONLY | selected=10 (5 SPAM + 5 NOT_SPAM) | chat={defender["model"]} | tool={defender["tool_id"]}')
    print('Endpoint:',defender['url'],'| Output:',output)
    print('Baseline classification only; no rewrite-attacker, GAN, Clef, or d3 calls.' if args.attack=='none' else
          f'Attack pilot: {args.attack.upper()}; only eligible PPLX spam is rewritten. Clef/d3 are not called.')
    if defender.get('revision_status')=='UNVERIFIED_WEIGHTS':
        print('Connectivity test only: backend weights are unverified; metadata hashes are not weight revisions.')
    if args.dry_run:
        # Reuse local catalog hashes for offline validation, without claiming
        # current server availability. All remaining identity fields must match.
        for folder in (baseline_output,output):
            saved_path=folder/'pplx/run.json'
            if not saved_path.exists():continue
            saved=json.loads(saved_path.read_text(encoding='utf-8'))
            pairs=[(defender,saved.get('defender',{}))]
            if folder==output and args.attack=='llm':pairs.append((config['attacker'],saved.get('attack_contract',{}).get('attacker',{})))
            for current,previous in pairs:
                without_hash=lambda value:{k:v for k,v in value.items() if k!='pilot_metadata_sha256'}
                if without_hash(current)==without_hash(previous) and 'pilot_metadata_sha256' in previous:
                    current['pilot_metadata_sha256']=previous['pilot_metadata_sha256']
        print('Offline check: saved catalog identity only; server availability is not checked.')
    if not args.dry_run:
        key_name=defender.get('key_env','')
        if key_name and not os.environ.get(key_name,'').strip():
            raise ValueError(f'Set {key_name} in {PROJECT_ENV} or export it in this terminal. The pilot reads the project-root .env automatically.')
        snapshot=resolve_route(defender,args.timeout,args.list_tools)
        if not args.list_tools:reuse_legacy_catalog_identity(defender,snapshot,[baseline_output,output])
        if not args.list_tools and args.attack=='llm':
            attacker=config['attacker'];url=attacker['url']
            if not url.endswith('/chat/completions'):raise ValueError('Attacker requires an OpenAI-compatible chat/completions endpoint')
            models=catalog_rows(get_json(url[:-len('/chat/completions')]+'/models',pipeline.credentials(attacker),args.timeout),'models')
            matches=[row for row in models if row['id']==attacker['model']]
            if len(matches)!=1:raise ValueError('Attacker model is not registered: '+attacker['model']+'. Check the server model ID.')
            attacker['pilot_metadata_sha256']=pipeline.fingerprint({k:matches[0][k] for k in ('id','digest','ollama','info') if k in matches[0]})
            print('Attacker catalog verified:',attacker['model'],'|',url)
        if args.check_server and config.get('rewrite_evaluator') and args.attack!='none':
            resolve_route(config['rewrite_evaluator'],args.timeout)
        if args.list_tools or args.check_server:
            if args.check_server and args.attack=='gan':
                from gan_engine import GanEngine
                GanEngine(config.get('gan',{}))
                print('GAN weights and Python runtime verified; no inference.')
            if args.check_server:print('Catalog check passed; actual tool execution is checked by the 10-email pilot.')
            return 0
    if config.get('rewrite_evaluator') and args.attack!='none':
        evaluator=config['rewrite_evaluator']
        pipeline.validate_endpoint(evaluator,live=False)
        if not evaluator.get('revision'):
            evaluator.update(revision='UNPINNED_VALIDITY_PILOT',revision_status='UNVERIFIED_WEIGHTS')
        if not args.dry_run:
            resolve_route(evaluator,args.timeout)
        else:
            saved_path=output/'pplx/run.json'
            if saved_path.exists():
                previous=json.loads(saved_path.read_text(encoding='utf-8')).get('attack_contract',{}).get('validity',{}).get('evaluator',{})
                without_hash=lambda v:{k:x for k,x in v.items() if k!='pilot_metadata_sha256'}
                if without_hash(evaluator)==without_hash(previous) and 'pilot_metadata_sha256' in previous:
                    evaluator['pilot_metadata_sha256']=previous['pilot_metadata_sha256']
    # Pass a resolved, secret-free profile in memory to the shared controller.
    # Clear only PPLX overrides so they cannot undo the verified runtime IDs.
    overrides={name:os.environ.pop(name) for name in ('PPLX_URL','PPLX_MODEL_ID','PPLX_CHECKPOINT_ID','PPLX_TOOL_ID',
               'ATTACKER_URL','ATTACKER_MODEL','ATTACKER_REVISION','ATTACKER_API_KEY_ENV') if name in os.environ}
    try:
        command=['--model','pplx','--pilot','--resume',
                 '--timeout',str(args.timeout),'--attempts',str(args.attempts),'--delay',str(args.delay)]
        if args.attack=='none':command.append('--baseline-only')
        else:command+=['--baseline-from',str(baseline_output)]
        if args.dry_run:command.append('--dry-run')
        if args.retry_errors:command.append('--retry-errors')
        if args.attack=='gan':
            import gan_pipeline
            result=gan_pipeline.main(command,config_override=config)
        else:result=pipeline.main(command,config_override=config)
    finally:os.environ.update(overrides)
    if not args.dry_run:
        print('Saved results:',output/'pplx/sample_results.csv')
        if result:print('Inspect events.csv and responses/. Chat-only labels are rejected; resume with --retry-errors after fixing the route.')
    return result


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:sys.exit(main())
    except KeyboardInterrupt:print('Pilot interrupted; committed results remain saved.');sys.exit(130)
    except (ValueError,OSError,KeyError) as exc:print('PPLX pilot error:',exc,file=sys.stderr);sys.exit(2)
