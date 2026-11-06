"""Offline behavioral tests for role separation, validity gates and resumability."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import adaptive_pipeline as core
import validity_pipeline as gated
import rewrite_validity as validity
import gan_engine
import evaluate_paper
import experiment_cli
import pilot_openwebui


def tool_response(params,label,tool='server:pplxdeciderflash'):
    name=tool+'/classify_with_pplx_decider'
    return {'sources':[{'source':{'name':name},'tool_result':True,
        'metadata':[{'source':name,'parameters':copy.deepcopy(params)}],
        'document':[json.dumps({'choice':label})]}]}


class ValidityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=core.ROOT/'scripts');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'decision_models').mkdir()
        (self.root/'decision_models/paper_prompts.json').write_bytes((core.ROOT/'decision_models/paper_prompts.json').read_bytes())
        self.root_patch=patch.object(core,'ROOT',self.root);self.root_patch.start();self.addCleanup(self.root_patch.stop)
        self.dataset=self.root/'data.csv'
        self.texts=['invalid1','invalid2','flip1','flip2','stubborn','uncertain','same','apierror']
        with self.dataset.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=['input_id','sample_id','kind','true_label','email_text']);w.writeheader()
            for text in self.texts+['ham']:
                w.writerow(dict(input_id=text,sample_id=text,kind='clean',true_label='NOT_SPAM' if text=='ham' else 'SPAM',email_text=text))
        self.evaluator=dict(transport='tool',url='http://mock/tool',model='google/gemma-3-27b-it',tool_id='server:pplxdeciderflash',revision='test',key_env='',role='pplx_validity',rubric_version=validity.VERSION)
        defender={k:v for k,v in self.evaluator.items() if k not in ('role','rubric_version')}
        self.config=dict(profile='experimenting',protocol='validity_gated_v1',dataset=str(self.dataset),
            output_dir=str(self.root/'experimenting/outputs/llm'),gan_output_dir=str(self.root/'experimenting/outputs/gan'),
            defenders={'pplx':defender},active_models=['pplx'],default_model='pplx',rewrite_evaluator=self.evaluator,
            attacker=dict(transport='openai',url='http://mock/generate',model='gemma',revision='test',key_env='',temperature=0,max_tokens=1024),gan={})
        self.calls=[];self.fail=False
        self.request_patch=patch.object(core.api,'request_json',side_effect=self.request)
        self.request_patch.start();self.addCleanup(self.request_patch.stop)

    def request(self,url,key,body,timeout,attempts):
        self.calls.append(copy.deepcopy(body))
        if url.endswith('/generate'):
            text=body['messages'][1]['content']
            generated=text if text=='same' else 'R2 '+text if text.startswith('R1 ') else 'R1 '+text
            return {'choices':[{'finish_reason':'stop','message':{'content':generated}}]},200,3,''
        params=json.loads(body['messages'][1]['content'])
        if set(params['criteria'])==set(validity.CATEGORIES):
            pair=json.loads(params['state']);original=pair['original_email'];candidate=pair['candidate_rewrite']
            if self.fail and original=='apierror' and params['instructions']==validity.INSTRUCTIONS+validity.RUBRIC['meaning']:
                raise RuntimeError('simulated evaluator timeout')
            label='FAIL' if original=='invalid1' or (original=='invalid2' and candidate.startswith('R2')) else 'UNCERTAIN' if original=='uncertain' else 'PASS'
        else:
            text=params['state']
            label='NOT_SPAM' if text=='ham' or (text.startswith('R1') and 'flip1' in text) or (text.startswith('R2') and 'flip2' in text) else 'SPAM'
        return tool_response(params,label),200,2,''

    def run_pipeline(self,kind='llm',*args):
        with contextlib.redirect_stdout(io.StringIO()):
            if kind=='gan':
                class Engine:
                    contract={'test':1}
                    def __init__(self,config):pass
                    def generate(self,text,sid):return text if text=='same' else ('R2 ' if ':step2' in sid else 'R1 ')+text
                with patch.object(gan_engine,'GanEngine',Engine),patch.object(gan_engine,'gan_contract',return_value={'test':1}):
                    return gated.main(['--model','pplx',*args],self.config,'gan')
            return gated.main(['--model','pplx',*args],self.config,'llm')

    def load(self,kind='llm'):
        folder=self.root/'experimenting/outputs'/kind/'pplx'
        saved=json.loads((folder/'run.json').read_text())
        ledger=gated.ValidityLedger(folder.parent,'pplx',saved['identities'],set(self.texts+['ham']))
        return folder,saved,ledger

    def assert_flow(self,kind):
        self.assertEqual(self.run_pipeline(kind),0)
        folder,saved,ledger=self.load(kind)
        self.assertEqual(ledger.ok('invalid1','validate1')['label'],'INVALID')
        self.assertIsNone(ledger.get('invalid1','classify1'));self.assertIsNone(ledger.get('invalid1','rewrite2'))
        self.assertEqual(ledger.ok('invalid2','validate2')['label'],'INVALID');self.assertIsNone(ledger.get('invalid2','classify2'))
        self.assertIsNone(ledger.get('flip1','rewrite2'));self.assertIsNone(ledger.get('same','classify1'))
        self.assertEqual(ledger.ok('uncertain','validate1')['label'],'UNCERTAIN')
        self.assertIsNone(ledger.get('ham','rewrite1'))
        metrics=json.loads((folder/'metrics.json').read_text())
        self.assertEqual(metrics['overall_valid_attack_success_rate'],2/8)
        self.assertEqual(metrics['steps']['step1']['INVALID'],1)
        self.assertEqual(metrics['steps']['step2']['INVALID'],1)
        self.assertEqual(metrics['steps']['step1']['NO_CHANGE'],1)
        self.assertEqual(metrics['steps']['step1']['UNCERTAIN'],1)
        for body in self.calls:
            if body['messages'][0]['content'].startswith('Invoke'):
                params=json.loads(body['messages'][1]['content'])
                if set(params['criteria'])==set(validity.CATEGORIES):
                    pair=json.loads(params['state']);self.assertIn(pair['original_email'],self.texts)
                    self.assertEqual(set(pair),{'original_email','candidate_rewrite'})
        before=len(self.calls);self.assertEqual(self.run_pipeline(kind),0);self.assertEqual(len(self.calls),before)
        exported=core.read_csv(folder/'step1_rewrites.csv')
        rejected=next(r for r in exported if r['sample_id']=='invalid1')
        self.assertEqual(rejected['classification_status'],'SKIPPED_INVALID')
        self.assertEqual(rejected['validity_coherence'],'FAIL')
        self.assertTrue(rejected['validation_time_sec'])
        self.assertEqual(len(list((folder/'rewrites/step1').glob('*.csv'))),8)
        with contextlib.redirect_stdout(io.StringIO()):
            loaded,rows,manifest=evaluate_paper.load_run(folder.parent,'pplx')
            evaluate_paper.evaluate_run(loaded,rows,manifest)
        self.assertEqual(len(self.calls),before)
        pairs=core.read_csv(folder/'evaluation/similarity_per_rewrite.csv')
        self.assertTrue(any(p['outcome']=='invalid' and p['validation_status']=='INVALID' for p in pairs))
        self.assertFalse(any(p['stage']=='final' and p['outcome']=='invalid' for p in pairs))
        self.assertEqual(json.loads((folder/'metrics.json').read_text())['overall_valid_attack_success_rate'],2/8)

    def test_llm_two_steps_gates_resume_evaluation(self):self.assert_flow('llm')
    def test_gan_two_steps_gates_resume_evaluation(self):self.assert_flow('gan')

    def test_evaluator_error_partial_cache_and_retry(self):
        self.fail=True;self.assertEqual(self.run_pipeline(),2)
        folder,saved,ledger=self.load()
        self.assertEqual(ledger.get('apierror','validate1')['status'],'ERROR')
        self.assertIsNone(ledger.get('apierror','classify1'))
        before=len(self.calls);self.run_pipeline();self.assertEqual(len(self.calls),before)
        self.fail=False;self.assertEqual(self.run_pipeline('llm','--retry-errors'),0)
        calls=[json.loads(c['messages'][1]['content']) for c in self.calls if c['messages'][0]['content'].startswith('Invoke')]
        checks=[p for p in calls if set(p['criteria'])==set(validity.CATEGORIES) and json.loads(p['state'])=={'original_email':'apierror','candidate_rewrite':'R1 apierror'}]
        self.assertEqual(sum(p['instructions']==validity.INSTRUCTIONS+validity.RUBRIC['purpose'] for p in checks),1)
        self.assertEqual(sum(p['instructions']==validity.INSTRUCTIONS+validity.RUBRIC['meaning'] for p in checks),2)

    def test_rubric_change_rejects_checkpoint(self):
        self.run_pipeline();before=len(self.calls)
        with patch.dict(validity.RUBRIC,{'meaning':'changed rubric'}):
            with self.assertRaisesRegex(ValueError,'Different dataset/model/prompt/checkpoint'):self.run_pipeline()
        self.assertEqual(len(self.calls),before)

    def test_main_rejects_pplx_defender(self):
        self.config['profile']='main'
        with self.assertRaisesRegex(ValueError,'Clef/d3 only'):self.run_pipeline()
        self.assertEqual(self.calls,[])

    def test_main_uses_only_clef_d3_classification_routes(self):
        self.config.update(profile='main',output_dir=str(self.root/'outputs/main'))
        self.config['defenders']={n:dict(transport='direct',url=f'http://mock/{n}',model=n,revision='test',key_env='') for n in ('clef','d3')}
        self.config['active_models']=['clef','d3']; routes=[]
        def request(url,key,body,timeout,attempts):
            routes.append(url)
            if url.endswith(('/clef','/d3')):return {'label':'SPAM'},200,2,''
            return self.request(url,key,body,timeout,attempts)
        with patch.object(core.api,'request_json',side_effect=request),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gated.main(['--model','all'],self.config),0)
        self.assertIn('http://mock/clef',routes);self.assertIn('http://mock/d3',routes)
        for body in self.calls:
            if body['messages'][0]['content'].startswith('Invoke'):
                self.assertEqual(set(json.loads(body['messages'][1]['content'])['criteria']),set(validity.CATEGORIES))

    def test_parser_rejects_chat_wrong_arguments_conflicts(self):
        params=validity.parameters('original','candidate','meaning')
        with self.assertRaises(ValueError):validity.parse_result({'choices':[{'message':{'content':'PASS'}}]},self.evaluator['tool_id'],params)
        response=tool_response(params,'PASS');response['sources'][0]['metadata'][0]['parameters']['state']='changed'
        with self.assertRaises(ValueError):validity.parse_result(response,self.evaluator['tool_id'],params)
        response=tool_response(params,'PASS');response['sources'][0]['document'].append(json.dumps({'label':'FAIL'}))
        with self.assertRaises(ValueError):validity.parse_result(response,self.evaluator['tool_id'],params)
        for label in validity.CATEGORIES:self.assertEqual(validity.parse_result(tool_response(params,label),self.evaluator['tool_id'],params),label)

    def test_gan_capture_never_returns_a_defender_label(self):
        engine=object.__new__(gan_engine.GanEngine)
        observed=[]
        def attack(text,sid,victim):
            observed.append('generated');victim.get_pred(['candidate']);observed.append('classified')
        engine.attack=attack
        self.assertEqual(engine.generate('original','sample:step1'),'candidate')
        self.assertEqual(observed,['generated'])

    def test_baseline_then_gated_run_reuses_classifications(self):
        self.assertEqual(self.run_pipeline('llm','--baseline-only'),0)
        before=len(self.calls)
        self.assertEqual(self.run_pipeline(),0)
        for body in self.calls[before:]:
            if body.get('tool_ids'):
                params=json.loads(body['messages'][1]['content'])
                if set(params['criteria'])=={'SPAM','NOT_SPAM'}:self.assertTrue(params['state'].startswith(('R1','R2')))

    def test_interruption_after_generation_resumes_without_regeneration(self):
        with patch.object(validity,'evaluate',side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_pipeline(),130)
        before=sum(not bool(c.get('tool_ids')) for c in self.calls)
        folder,saved,ledger=self.load()
        self.assertIsNotNone(ledger.ok('invalid1','rewrite1'))
        self.assertEqual(json.loads((folder/'checkpoint.json').read_text())['state'],'interrupted')
        self.assertEqual(self.run_pipeline(),0)
        first=[c for c in self.calls if not c.get('tool_ids') and c['messages'][1]['content']=='invalid1']
        self.assertEqual(len(first),1)

    def test_new_pilot_route_and_baseline_import_then_offline_resume(self):
        # Exercise the real pilot wrapper and dispatcher; no remote catalog or model calls.
        with self.dataset.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['input_id','sample_id','kind','true_label','email_text']);writer.writeheader()
            for i in range(10):
                text=self.texts[i] if i<5 else 'ham'+str(i)
                writer.writerow(dict(input_id=text,sample_id=text,kind='clean',true_label='SPAM' if i<5 else 'NOT_SPAM',email_text=text))
        self.config['attacker']['url']='http://mock/api/chat/completions'
        self.config['output_dir']=str(self.root/'experimenting/outputs/full')
        for name,folder in [('tool_pilot_output_dir','base'),('tool_llm_pilot_output_dir','llm')]:
            self.config[name]=str(self.root/'experimenting/outputs'/folder)
        path=self.root/'profile.json';path.write_text(json.dumps(self.config))
        original_request=self.request
        def request(url,key,body,timeout,attempts):
            if not body.get('tool_ids'):url='http://mock/generate'
            return original_request(url,key,body,timeout,attempts)
        def resolve(config,*args):config['pilot_metadata_sha256']='mock-catalog';return {}
        with patch.object(pilot_openwebui,'resolve_route',side_effect=resolve),patch.object(pilot_openwebui,'get_json',return_value={'data':[{'id':'gemma'}]}),patch.object(core.api,'request_json',side_effect=request),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot_openwebui.main(['--config',str(path),'--profile-routes']),0)
            self.assertEqual(len(self.calls),10)
            self.assertEqual(pilot_openwebui.main(['--config',str(path),'--profile-routes','--attack','llm']),0)
            before=len(self.calls)
            self.assertEqual(pilot_openwebui.main(['--config',str(path),'--profile-routes','--attack','llm','--dry-run']),0)
            self.assertEqual(pilot_openwebui.main(['--config',str(path),'--profile-routes','--attack','llm']),0)
            self.assertEqual(len(self.calls),before)
        events=core.read_csv(self.root/'experimenting/outputs/llm/pplx/events.csv')
        self.assertEqual(sum(e['source']=='baseline_reuse' for e in events),10)


if __name__=='__main__':unittest.main()
