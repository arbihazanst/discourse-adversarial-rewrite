"""PPLX-only pilot against a local Open WebUI mock; no production calls."""
import contextlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import adaptive_pipeline as core
import experiment_cli
import pilot_openwebui as pilot
import run_decision_benchmark as api


class Server(BaseHTTPRequestHandler):
    posts=[];gets=[];chat_only=False;wrong_tool=False;missing_tool=False;catalog_stamp=0
    def log_message(self,*args):pass
    def reply(self,value):
        body=json.dumps(value).encode();self.send_response(200)
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        self.gets.append(self.path)
        if self.headers.get('Authorization')!='Bearer test-only':
            self.send_response(401);self.end_headers();return
        if self.path=='/api/models':self.reply({'data':[{'id':'mock/orchestrator','digest':'mock-digest'},{'id':'google/gemma-3-27b-it'}]})
        elif self.path=='/api/v1/tools/':
            type(self).catalog_stamp+=1
            self.reply([] if self.missing_tool else [{'id':'pplx_decider','name':'PPLX Decider','updated_at':self.catalog_stamp}])
        else:self.send_response(404);self.end_headers()
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.posts.append((self.path,body))
        if body['model']=='google/gemma-3-27b-it' and not body.get('tool_ids'):
            self.reply({'choices':[{'finish_reason':'stop','message':{'content':'Neutral rewrite: '+body['messages'][1]['content']}}]});return
        result={'choices':[{'message':{'content':'{"label":"NOT_SPAM"}'}}]}
        if not self.chat_only:
            result['sources']=[{'source':{'type':'tool','id':'unrelated' if self.wrong_tool else 'pplx_decider'},
                                'metadata':[{'parameters':json.loads(body['messages'][1]['content'])}],
                                'document':[json.dumps({'label':'SPAM','probabilities':{'SPAM':0.8,'NOT_SPAM':0.2}})]}]
        self.reply(result)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='pilot test ',dir=core.ROOT/'experimenting');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'pilot';self.profile=self.root/'config.json'
        config=json.loads((core.ROOT/'experimenting/adaptive_config.json').read_text())
        config['defenders']['pplx'].update(model='mock/orchestrator',tool_id='pplx-decider',revision='')
        # Deliberately keep inactive unreachable models to prove isolation.
        config['defenders']['clef']={'transport':'direct','url':'http://unreachable.invalid','model':'clef','revision':'','key_env':''}
        config['tool_pilot_output_dir']=str(self.output)
        self.profile.write_text(json.dumps(config))
        env={k:'' for k in os.environ if k.startswith(('PPLX_','ATTACKER_'))}
        env['OWUI_API_KEY']='test-only'
        env['DECISION_ENV_FILE']=str(self.root/'absent.env')
        p=patch.dict(os.environ,env);p.start();self.addCleanup(p.stop)
        Server.posts=[];Server.gets=[];Server.chat_only=False;Server.wrong_tool=False;Server.missing_tool=False
        self.server=HTTPServer(('127.0.0.1',0),Server)
        thread=threading.Thread(target=self.server.serve_forever,daemon=True);thread.start()
        def stop():self.server.shutdown();self.server.server_close();thread.join()
        self.addCleanup(stop)
        self.url=f'http://127.0.0.1:{self.server.server_port}'
        config['attacker']['url']=self.url+'/api/chat/completions'
        self.profile.write_text(json.dumps(config))

    def run_pilot(self,*args):
        with contextlib.redirect_stdout(io.StringIO()):
            return pilot.main(['--config',str(self.profile),'--base-url',self.url,*args])

    def test_ten_originals_verified_tool_only_and_resume(self):
        self.assertEqual(self.run_pilot(),0)
        self.assertEqual(len(Server.posts),10)
        self.assertEqual(set(Server.gets),{'/api/models','/api/v1/tools/'})
        for path,payload in Server.posts:
            self.assertEqual(path,'/api/chat/completions')
            self.assertEqual(payload['model'],'mock/orchestrator')
            self.assertEqual(payload['tool_ids'],['pplx_decider'])
            self.assertEqual(payload['params'],{'function_calling':'legacy'})
            self.assertIn('exactly once',payload['messages'][0]['content'])
        rows=core.read_csv(self.output/'pplx/sample_results.csv')
        self.assertEqual(len(rows),10)
        self.assertEqual(sum(r['true_label']=='SPAM' for r in rows),5)
        events=core.read_csv(self.output/'pplx/events.csv')
        self.assertEqual({r['stage'] for r in events},{'baseline'})
        self.assertEqual({r['label'] for r in events},{'SPAM'}) # Ignore opposite chat label.
        self.assertTrue(all(float(r['latency_ms'])>=0 for r in events))
        saved=json.loads((self.output/'pplx/run.json').read_text())
        self.assertEqual(saved['defender']['revision_status'],'UNVERIFIED_WEIGHTS')
        self.assertNotIn('test-only',(self.output/'pplx/run.json').read_text())
        self.assertFalse((self.output/'clef').exists())
        self.assertEqual(self.run_pilot('--resume'),0)
        self.assertEqual(len(Server.posts),10)

    def test_chat_only_errors_then_explicit_retry(self):
        Server.chat_only=True
        self.assertEqual(self.run_pilot(),2)
        self.assertEqual(self.run_pilot('--resume'),2)
        self.assertEqual(len(Server.posts),10)
        events=core.read_csv(self.output/'pplx/events.csv')
        self.assertTrue(all(r['status']=='ERROR' and not r['label'] for r in events))
        Server.chat_only=False
        self.assertEqual(self.run_pilot('--retry-errors'),0)
        self.assertEqual(len(Server.posts),20)

    def test_mismatched_tool_provenance_is_rejected(self):
        Server.wrong_tool=True
        self.assertEqual(self.run_pilot(),2)
        self.assertTrue(all(r['status']=='ERROR' for r in core.read_csv(self.output/'pplx/events.csv')))

    def test_catalog_failure_before_classification(self):
        Server.missing_tool=True
        with self.assertRaisesRegex(ValueError,'Available IDs'):self.run_pilot()
        self.assertEqual(Server.posts,[]);self.assertFalse(self.output.exists())

    def test_offline_check_needs_no_key_or_network_and_check_server_needs_no_inference(self):
        with patch.dict(os.environ,{'OWUI_API_KEY':''}),patch.object(pilot,'get_json',side_effect=AssertionError('Network in dry-run')):
            self.assertEqual(self.run_pilot('--dry-run'),0)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.run_pilot('--check-server'),0)
        self.assertEqual(Server.posts,[]);self.assertFalse(self.output.exists())
        with patch.dict(os.environ,{'OWUI_API_KEY':''}),self.assertRaisesRegex(ValueError,'OWUI_API_KEY'):
            self.run_pilot()

    def test_tool_discovery_and_output_isolation(self):
        self.assertEqual(self.run_pilot('--list-tools'),0)
        self.assertEqual(Server.posts,[])
        with self.assertRaisesRegex(ValueError,'inside experimenting'):
            self.run_pilot('--output-dir',str(core.ROOT/'outputs/should_not_exist'),'--dry-run')
        with self.assertRaises(ValueError):
            pilot.select_tool([{'id':'a','name':'PPLX Decider'},{'id':'b','name':'PPLX Decider'}],'server:pplxdeciderflash')

    def test_active_models_and_shared_payload(self):
        config=json.loads((core.ROOT/'decision_models/adaptive_main.json').read_text())
        self.assertEqual(core.selected_models(config,'all'),['pplx'])
        self.assertEqual(core.selected_models(config,'clef'),['clef'])
        defender=config['defenders']['pplx'];text='Full email\nSubject and body'
        self.assertEqual(core.defender_payload(defender,text),api.tool_payload(defender['model'],defender['tool_id'],text))
        with patch.dict(os.environ,{'PPLX_TOOL_ID':'old'}):
            self.assertNotIn('PPLX_TOOL_ID',experiment_cli.child_environment(config))

    def test_both_profiles_use_clef_d3_pipes_and_keep_pplx_active(self):
        for path in ('decision_models/adaptive_main.json','experimenting/adaptive_config.json'):
            config=json.loads((core.ROOT/path).read_text())
            self.assertEqual(core.selected_models(config,'all'),['pplx'])
            for name in ('clef','d3'):
                defender=config['defenders'][name]
                self.assertEqual(defender['transport'],'openwebui')
                self.assertEqual(defender['url'],'https://llms.si-lab.org/api/chat/completions')
                payload=core.defender_payload(defender,'Original email')
                self.assertEqual(payload,{'model':'decision_spam.'+name,'messages':[{'role':'user','content':'Original email'}],'stream':False})

    def test_menu_connection_pilot_uses_profile(self):
        with patch('builtins.input',side_effect=['1','1','1','0']),patch.object(experiment_cli,'run_command',return_value=0) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment_cli.main(['--config',str(core.ROOT/'decision_models/adaptive_main.json')]),0)
        self.assertIn(str(core.ROOT/'scripts/pilot_openwebui.py'),run.call_args.args[0])

    def test_llm_pilot_reuses_baseline_and_two_paper_prompts(self):
        self.assertEqual(self.run_pilot(),0)
        output=self.root/'llm_attack'
        self.assertEqual(self.run_pilot('--attack','llm','--output-dir',str(output)),0)
        calls=[body for _,body in Server.posts if not body.get('tool_ids')]
        self.assertEqual(len(calls),10) # Five eligible, two rewrites each.
        prompts=json.loads((core.ROOT/'decision_models/paper_prompts.json').read_text(encoding='utf-8'))
        self.assertEqual(calls[0]['messages'][0]['content'],prompts['ATTACKER_PROMPT_1'])
        self.assertIn('FIRST REWRITE',calls[-1]['messages'][0]['content'])
        events=core.read_csv(output/'pplx/events.csv')
        self.assertEqual(sum(e['source']=='baseline_reuse' for e in events),10)
        self.assertEqual(len(Server.posts),30) # 10 baseline + 10 generation + 10 classify.
        self.assertEqual(self.run_pilot('--attack','llm','--output-dir',str(output),'--resume'),0)
        self.assertEqual(len(Server.posts),30)

    def test_recovery_uses_saved_sources_without_requests(self):
        import reparse_saved_tools
        with patch.object(api,'extract',side_effect=ValueError('old parser format mismatch')):
            self.assertEqual(self.run_pilot(),2)
        with patch.object(api,'request_json',side_effect=AssertionError('Network during recovery')):
            self.assertEqual(reparse_saved_tools.recover(self.output),10)
        self.assertEqual(self.run_pilot('--resume'),0)
        self.assertEqual(len(Server.posts),10)

    def test_gan_pilot_reuses_baseline_without_attacker_chat(self):
        import gan_pipeline as gan
        from test_gan_pipeline import FakeEngine, contract
        self.assertEqual(self.run_pilot(),0)
        output=self.root/'gan_attack'
        with patch.object(gan,'GanEngine',FakeEngine),patch.object(gan,'gan_contract',contract):
            self.assertEqual(self.run_pilot('--attack','gan','--output-dir',str(output)),0)
            events=core.read_csv(output/'pplx/events.csv')
            self.assertEqual(sum(e['source']=='baseline_reuse' for e in events),10)
            self.assertEqual(len(list((output/'pplx/rewrites/gan').glob('*.csv'))),5)
            self.assertEqual(len(Server.posts),20) # Ten baseline + five probes + five final classifications.
            self.assertTrue(all(body.get('tool_ids') for _,body in Server.posts))
            self.assertEqual(self.run_pilot('--attack','gan','--output-dir',str(output),'--resume'),0)
            self.assertEqual(len(Server.posts),20)

    def test_external_server_provenance_shape(self):
        import copy
        name='server:pplxdeciderflash/classify_with_pplx_decider'
        source={'source':{'name':name},'tool_result':True,'metadata':[{'source':name,'parameters':{'state':'email'}}],
                'document':[json.dumps({'answers':{'decision':{'choice':'spam','probabilities':{'spam':.9,'not_spam':.1}}}})]}
        self.assertEqual(api.extract({'sources':[source]},'tool','server:pplxdeciderflash'),('SPAM','0.9','0.1'))
        for change in ('flag','namespace','metadata'):
            bad=copy.deepcopy(source)
            if change=='flag':bad['tool_result']=False
            elif change=='namespace':bad['source']['name']=name.replace('pplxdeciderflash/','pplxdeciderflash_other/')
            else:bad['metadata'][0]['source']='different'
            with self.assertRaises(ValueError):api.extract({'sources':[bad]},'tool','server:pplxdeciderflash')

    def test_fixed_tool_parameters_reject_changed_email_and_categories(self):
        import copy
        name='server:pplxdeciderflash/classify_with_pplx_decider'
        expected=api.tool_parameters('original email')
        source={'source':{'name':name},'tool_result':True,
                'metadata':[{'source':name,'parameters':expected}], 'document':['{"label":"SPAM"}']}
        self.assertEqual(api.extract({'sources':[source]},'tool','server:pplxdeciderflash','original email')[0],'SPAM')
        for key,value in [('state','truncated'),('criteria',{'billing':'billing','spam':'spam'}),('instructions','new task')]:
            bad=copy.deepcopy(source);bad['metadata'][0]['parameters'][key]=value
            with self.assertRaisesRegex(ValueError,'fixed SPAM/NOT_SPAM'):
                api.extract({'sources':[bad]},'tool','server:pplxdeciderflash','original email')

    def test_legacy_catalog_compatibility_requires_exact_hash_proof(self):
        import copy,csv
        snapshot={'model':{'id':'mock'},'tool':{'id':'tool','name':'Tool'}}
        old=copy.deepcopy(snapshot);old['tool']['updated_at']=990
        digest=core.fingerprint(old)
        defender={'model':'mock','pilot_metadata_sha256':core.fingerprint(snapshot)}
        folder=self.root/'legacy';(folder/'pplx').mkdir(parents=True)
        (folder/'pplx/run.json').write_text(json.dumps({'defender':{**defender,'pilot_metadata_sha256':digest}}))
        (folder/'pplx/response.json').write_text(json.dumps({'response':{'created':1000}}))
        with (folder/'pplx/events.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['artifact']);w.writeheader();w.writerow({'artifact':'pplx/response.json'})
        pilot.reuse_legacy_catalog_identity(defender,snapshot,[folder])
        self.assertEqual(defender['pilot_metadata_sha256'],digest)
        self.assertTrue((folder/'pplx/pilot_catalog_compatibility.json').exists())
        changed=copy.deepcopy(snapshot);changed['tool']['name']='Changed tool'
        defender['pilot_metadata_sha256']=core.fingerprint(changed)
        pilot.reuse_legacy_catalog_identity(defender,changed,[folder])
        self.assertNotEqual(defender['pilot_metadata_sha256'],digest)

    def test_pilot_reads_dotenv_from_other_working_directory(self):
        dotenv=self.root/'pilot.env';dotenv.write_text('OWUI_API_KEY=test-only\nPPLX_CHAT_MODEL=mock/orchestrator\n',encoding='utf-8')
        env=dict(os.environ,DECISION_ENV_FILE=str(dotenv),OWUI_API_KEY='',PPLX_URL='',PPLX_MODEL_ID='')
        result=subprocess.run([sys.executable,str(core.ROOT/'scripts/pilot_openwebui.py'),'--config',str(self.profile),
                               '--base-url',self.url,'--check-server'],cwd=self.root,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotIn('test-only',result.stdout+result.stderr)
        self.assertEqual(Server.posts,[])



    def test_offline_resume_and_custom_attack_output(self):
        self.assertEqual(self.run_pilot(),0)
        config=json.loads(self.profile.read_text())
        config['defenders']['pplx']['tool_id']='pplx_decider' # Canonical ID; offline checks cannot resolve server aliases.
        config['tool_llm_pilot_output_dir']=str(self.root/'configured_llm')
        self.profile.write_text(json.dumps(config))
        self.assertEqual(self.run_pilot('--attack','llm'),0)
        before=len(Server.posts)
        self.assertEqual(self.run_pilot('--attack','llm','--dry-run'),0)
        self.assertEqual(len(Server.posts),before)
        self.assertTrue((self.root/'configured_llm/pplx/step1_rewrites.csv').exists())


if __name__=='__main__':unittest.main()
