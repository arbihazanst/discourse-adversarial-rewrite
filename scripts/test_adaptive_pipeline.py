"""End-to-end protocol tests using local mock HTTP endpoints; no live models."""
import contextlib
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
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

import adaptive_pipeline as pipeline


class Server(BaseHTTPRequestHandler):
    calls=[]
    malformed=False
    truncated=False

    def log_message(self,*args):pass

    def do_GET(self):
        self.reply({'models':[{'name':'clef:27b','digest':'digest-clef'},
                              {'name':'server:pplxdeciderflash:latest','digest':'digest-pplx'},
                              {'name':'gemma3:27b','digest':'digest-gemma'}]})

    def reply(self,data):
        b=json.dumps(data).encode();self.send_response(200)
        self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)

    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.calls.append((self.path,body))
        if self.path in {'/chat','/api/chat'}:
            first=body['messages'][1]['content']
            second='FIRST REWRITE' in body['messages'][0]['content']
            text=('second '+first) if second else ('first '+first)
            self.reply({'model':body['model'],'message':{'content':text},'done':True,'done_reason':'stop'}
                       if self.path=='/api/chat' else
                       {'choices':[{'finish_reason':'length' if self.truncated else 'stop',
                                   'message':{'content':text}}]});return
        text=body.get('state') or body.get('email') or body.get('messages',[{'content':''}])[0]['content']
        if self.path=='/tool':text=json.loads(body['messages'][1]['content'])['state']
        label='NOT_SPAM' if 'missed' in text or (text.startswith('first') and 'one' in text) or (text.startswith('second') and 'two' in text) else 'SPAM'
        if self.malformed and text=='spam-two':self.reply({'response':'not structured'});return
        if self.path=='/v1/systemone':
            self.reply({'model':body['model'],'answers':{'spam':{'choice':label}}})
        elif self.path=='/pipe':
            self.reply({'choices':[{'message':{'content':json.dumps({'label':label})}}]})
        elif self.path=='/tool':
            self.reply({'sources':[{'type':'tool','id':'verified-tool','metadata':[{'parameters':json.loads(body['messages'][1]['content'])}],'content':{'label':label}}]})
        else:self.reply({'label':label})


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='test_adaptive_',dir=pipeline.ROOT/'scripts');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.out=self.root/'out'
        self.dataset=self.root/'data.csv';self.bank=self.root/'bank.csv'
        with self.dataset.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['input_id','sample_id','kind','true_label','email_text']);w.writeheader()
            for i,text in enumerate(['spam-one','spam-two','spam-stubborn','spam-missed','ham-falsepositive']):
                w.writerow(dict(input_id=str(i)+':clean',sample_id=str(i),kind='clean',
                                true_label='NOT_SPAM' if text.startswith('ham') else 'SPAM',email_text=text))
        self.bank.write_text('sample_id,original_text_sha256,rewrite_text,source_csv\n',encoding='utf-8')
        Server.calls=[];Server.malformed=False;Server.truncated=False
        server=HTTPServer(('127.0.0.1',0),Server)
        self.thread=threading.Thread(target=server.serve_forever,daemon=True);self.thread.start()
        def close():server.shutdown();server.server_close();self.thread.join()
        self.addCleanup(close)
        self.url=f'http://127.0.0.1:{server.server_port}'
        self.config=dict(profile='main',dataset=str(self.dataset),step1_bank=str(self.bank),output_dir=str(self.out),
            attacker=dict(transport='openai',url=self.url+'/chat',model='gemma3:27b',revision='test-attacker',key_env='',temperature=0,max_tokens=1024),
            defenders={'clef':dict(transport='direct',url=self.url+'/decision',model='clef',revision='test-clef',key_env='')})
        self.configpath=self.root/'config.json'
        self.env=patch.dict(os.environ,{k:'' for k in ['ATTACKER_URL','ATTACKER_MODEL','ATTACKER_REVISION','ATTACKER_API_KEY_ENV',
            'CLEF_URL','CLEF_MODEL_ID','CLEF_CHECKPOINT_ID','PPLX_URL','PPLX_MODEL_ID','PPLX_CHECKPOINT_ID']})
        self.env.start();self.addCleanup(self.env.stop)

    def run_pipeline(self,*args):
        self.configpath.write_text(json.dumps(self.config),encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.main(['--config',str(self.configpath),'--attempts','1',*args])

    def metrics(self,model='clef'):
        return json.loads((self.out/model/'metrics.json').read_text())

    def test_full_protocol_ham_exclusion_two_attempts_and_resume(self):
        self.assertEqual(self.run_pipeline(),0)
        m=self.metrics()
        self.assertEqual((m['eligible'],m['step1_flips'],m['step2_additional_flips'],m['still_spam']),(3,1,1,1))
        self.assertEqual((m['FP'],m['FN']),(1,1))
        self.assertEqual(len([x for x in Server.calls if x[0]=='/chat']),5)
        self.assertEqual(len([x for x in Server.calls if x[0]=='/decision']),10)
        before=len(Server.calls)
        (self.out/'clef/checkpoint.json').write_text('corrupt',encoding='utf-8')
        self.assertEqual(self.run_pipeline('--resume'),0)
        self.assertEqual(len(Server.calls),before)
        self.assertEqual(json.loads((self.out/'clef/checkpoint.json').read_text())['state'],'complete')

    def test_baseline_then_full_and_limit_expansion(self):
        self.config['attacker']['revision']=''
        self.assertEqual(self.run_pipeline('--baseline-only','--limit','2'),0)
        self.assertEqual(len(Server.calls),2)
        self.config['attacker']['revision']='newly-recorded-revision'
        self.assertEqual(self.run_pipeline(),0)
        self.assertEqual(len([b for p,b in Server.calls if p=='/decision' and b['state']=='spam-one']),1)
        self.assertEqual(self.metrics()['selected_rows'],5)

    def test_classification_seconds_exclude_rewrites_and_survive_resume(self):
        request=pipeline.api.request_json
        def timed_request(url,*args,**kwargs):
            response,status,_,digest=request(url,*args,**kwargs)
            return response,status,9000 if url.endswith('/chat') else 1250,digest
        with patch.object(pipeline.api,'request_json',side_effect=timed_request):
            self.assertEqual(self.run_pipeline(),0)
        path=self.out/'clef/sample_results.csv'
        rows=pipeline.read_csv(path)
        self.assertEqual(rows[0]['baseline_time_sec'],'1.25')
        self.assertEqual(rows[0]['step1_time_sec'],'1.25')
        self.assertEqual(rows[0]['step2_time_sec'],'')
        self.assertEqual(rows[0]['classification_time_sec'],'2.5')
        self.assertEqual(rows[1]['step2_time_sec'],'1.25')
        self.assertEqual(rows[1]['classification_time_sec'],'3.75')
        self.assertEqual(rows[3]['step1_time_sec'],'')
        self.assertEqual(rows[3]['classification_time_sec'],'1.25')
        self.assertEqual(self.metrics()['classification_timed_requests'],10)
        self.assertEqual(self.metrics()['classification_total_time_sec'],12.5)
        self.assertEqual(self.metrics()['classification_mean_time_sec'],1.25)
        with patch.object(pipeline.api,'request_json',side_effect=AssertionError('Repeated inference')):
            self.assertEqual(self.run_pipeline('--resume'),0)
        self.assertEqual(pipeline.read_csv(path),rows)
        self.assertEqual(self.metrics()['classification_total_time_sec'],12.5)

    def test_errors_stay_unresolved_until_retry(self):
        Server.malformed=True
        self.assertEqual(self.run_pipeline(),2)
        self.assertEqual(self.metrics()['baseline_errors'],1)
        before=len(Server.calls);Server.malformed=False
        self.assertEqual(self.run_pipeline(),2)
        self.assertEqual(len(Server.calls),before)
        self.assertEqual(self.run_pipeline('--retry-errors'),0)
        self.assertEqual(self.metrics()['baseline_errors'],0)

    @unittest.skipUnless(os.name=='nt','Windows sharing violations')
    def test_atomic_checkpoint_retries_transient_windows_lock(self):
        target=self.root/'checkpoint.json';pipeline.atomic_json(target,{'old':True})
        denied=PermissionError('temporary sharing violation');denied.winerror=32
        real=pipeline.os.replace;attempts=[]
        def replace(source,destination):
            attempts.append(destination)
            if len(attempts)<3:
                self.assertEqual(json.loads(target.read_text()),{'old':True})
                raise denied
            return real(source,destination)
        with patch.object(pipeline.os,'replace',side_effect=replace),patch.object(pipeline.time,'sleep'):
            pipeline.atomic_json(target,{'new':True})
        self.assertEqual(len(attempts),3)
        self.assertEqual(json.loads(target.read_text()),{'new':True})
        self.assertEqual(list(self.root.glob('checkpoint.json.*.tmp')),[])

    def test_truncated_generation_not_classified_as_a_rewrite(self):
        Server.truncated=True
        self.assertEqual(self.run_pipeline(),2)
        self.assertEqual(self.metrics()['attack_unresolved'],3)
        self.assertEqual(len([x for x in Server.calls if x[0]=='/decision']),5)

    def test_interrupted_after_generation_resumes_without_regenerating(self):
        commit=pipeline.Ledger.commit
        def interrupted(obj,event,artifact,selected):
            commit(obj,event,artifact,selected)
            if event['stage']=='rewrite1':raise KeyboardInterrupt
        with patch.object(pipeline.Ledger,'commit',interrupted):
            self.assertEqual(self.run_pipeline(),130)
        self.assertEqual(self.run_pipeline('--resume'),0)
        first=[b for path,b in Server.calls if path=='/chat' and b['messages'][1]['content']=='spam-one']
        self.assertEqual(len(first),1)

    def test_shared_attacker_cache_keeps_identical_requests_across_defenders(self):
        self.config['defenders']['pplx']={**self.config['defenders']['clef'],'model':'pplx','revision':'test-pplx',
                                          'template':{'email':'__EMAIL_TEXT__'}}
        self.assertEqual(self.run_pipeline(),0)
        self.assertEqual(len([x for x in Server.calls if x[0]=='/chat']),5)
        self.assertEqual(self.metrics('pplx')['observed_flips'],2)

    def test_identity_and_artifact_corruption_rejected(self):
        self.assertEqual(self.run_pipeline(),0)
        self.config['attacker']['temperature']=0.3
        with self.assertRaisesRegex(ValueError,'Different dataset'):
            self.run_pipeline()
        self.config['attacker']['temperature']=0
        response=next((self.out/'clef/responses').glob('*.json'))
        response.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            self.run_pipeline()

    def test_historical_reuse_is_explicit_and_hash_checked(self):
        import hashlib
        with self.bank.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['sample_id','original_text_sha256','rewrite_text','source_csv']);w.writeheader()
            w.writerow(dict(sample_id='0',original_text_sha256=hashlib.sha256(b'spam-one').hexdigest(),
                            rewrite_text='first spam-one',source_csv='old.csv'))
        self.assertEqual(self.run_pipeline('--reuse-step1'),0)
        self.assertEqual(len([x for x in Server.calls if x[0]=='/chat']),4)
        with (self.out/'clef/events.csv').open(newline='',encoding='utf-8') as f:
            self.assertTrue(any(r['source']=='historical_unverified' for r in csv.DictReader(f)))

    def test_dry_run_offline_and_all_model_preflight(self):
        with patch.object(pipeline,'urlopen',side_effect=AssertionError('network')):
            self.assertEqual(self.run_pipeline('--dry-run'),0)
        self.assertFalse(self.out.exists())
        self.config['defenders']['pplx']={**self.config['defenders']['clef'],'model':'', 'transport':'openwebui'}
        with self.assertRaisesRegex(ValueError,'model/Pipe'):
            self.run_pipeline()
        self.assertEqual(Server.calls,[])
        self.config['defenders']['pplx']['transport']='direct'
        with self.assertRaisesRegex(ValueError,'verified native template'):
            self.run_pipeline('--dry-run')
        self.assertEqual(Server.calls,[])

    def test_ollama_pipe_tool_and_native_template_adapters(self):
        for transport,path in [('ollama','/v1/systemone'),('openwebui','/pipe'),('tool','/tool')]:
            with self.subTest(transport=transport):
                self.config['output_dir']=str(self.root/transport)
                self.config['defenders']['clef']=dict(transport=transport,url=self.url+path,
                    model='clef:27b' if transport=='ollama' else 'verified-model',
                    revision='' if transport=='ollama' else 'revision',key_env='',tool_id='verified-tool')
                self.assertEqual(self.run_pipeline('--baseline-only'),0)
        self.assertEqual(pipeline.defender_payload({'transport':'direct','template':{'email':'__EMAIL_TEXT__'}},'x'),{'email':'x'})

    def test_paper_data_and_prompts_are_frozen(self):
        rows=pipeline.api.load_dataset(pipeline.ROOT/'dataset/paper_baseline3000.csv',{'clean'})
        self.assertEqual(len(rows),3000)
        self.assertEqual(sum(r['true_label']=='SPAM' for r in rows),1500)
        self.assertEqual(len({r['sample_id'] for r in rows}),3000)
        original=pipeline.read_csv(pipeline.ROOT/'analysis/ceas_match/baseline_recovery/CEAS_08_gpt_oss_latest_transparent_20260220_185010.csv')
        self.assertEqual([r['email_text'] for r in rows],[r['text'] for r in original])
        prompts=json.loads((pipeline.ROOT/'decision_models/paper_prompts.json').read_text())
        self.assertIn('{original_email}',prompts['ATTACKER_PROMPT_2_TEMPLATE'])
        self.assertIn('{first_modified}',prompts['ATTACKER_PROMPT_2_TEMPLATE'])

    def test_paper_rates_and_separate_rewrite_csvs(self):
        self.assertEqual(self.run_pipeline(),0)
        m=self.metrics()
        self.assertEqual(m['spam_precision'],.75)
        self.assertEqual(m['spam_f1'],.75)
        self.assertEqual(m['macro_precision'],.375)
        self.assertEqual(m['macro_recall'],.375)
        self.assertEqual(m['weighted_recall'],.6)
        self.assertEqual(m['weighted_f1'],.6)
        self.assertAlmostEqual(m['paper_step1_asr_pct'],100/3)
        self.assertEqual(m['paper_step2_gain_pct'],50)
        self.assertAlmostEqual(m['paper_final_asr_pct'],200/3)
        self.assertAlmostEqual(m['paper_robust_accuracy_pct'],100/3)
        first=pipeline.read_csv(self.out/'clef/step1_rewrites.csv')
        second=pipeline.read_csv(self.out/'clef/step2_rewrites.csv')
        self.assertEqual(len(first),3);self.assertEqual(len(second),2)
        self.assertEqual(first[0]['original_text'],'spam-one')
        self.assertEqual(first[0]['rewrite_text'],'first spam-one')
        self.assertEqual(first[0]['predicted_label'],'NOT_SPAM')
        self.assertEqual(second[0]['first_rewrite_text'],'first spam-two')
        self.assertEqual(len(list((self.out/'clef/rewrites/step1').glob('*.csv'))),3)
        self.assertEqual(len(list((self.out/'clef/rewrites/step2').glob('*.csv'))),2)

    def test_evaluate_saved_run_needs_no_credentials_or_model_calls(self):
        import evaluate_paper
        self.assertEqual(self.run_pipeline(),0)
        before=(self.out/'clef/events.csv').read_bytes()
        with patch.object(pipeline.api,'request_json',side_effect=AssertionError('inference')):
            self.assertEqual(evaluate_paper.main(['--run-dir',str(self.out),'--lexical-only']),0)
        pairs=pipeline.read_csv(self.out/'clef/evaluation/similarity_per_rewrite.csv')
        self.assertEqual(sum(r['stage']=='step1' for r in pairs),3)
        self.assertEqual(sum(r['stage']=='step2' for r in pairs),2)
        self.assertEqual(sum(r['stage']=='final' for r in pairs),3)
        self.assertTrue(all(r['sbert']=='' for r in pairs))
        self.assertTrue(all(r['sbert_status']=='NOT_COMPUTED' for r in pairs))
        self.assertEqual((self.out/'clef/events.csv').read_bytes(),before)
        with patch.object(evaluate_paper.metrics,'lexical_scores',side_effect=AssertionError('Repeated evaluation')):
            self.assertEqual(evaluate_paper.main(['--run-dir',str(self.out),'--lexical-only']),0)
        self.dataset.write_text(self.dataset.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            evaluate_paper.main(['--run-dir',str(self.out),'--lexical-only'])

    def test_dynamic_defender_key_and_ollama_attacker(self):
        self.config['defenders']={'new_defender':self.config['defenders']['clef']}
        self.config['attacker'].update(transport='ollama_chat',url=self.url+'/api/chat',revision='')
        self.assertEqual(self.run_pipeline('--model','new_defender'),0)
        self.assertEqual(self.metrics('new_defender')['observed_flips'],2)
        saved=json.loads((self.out/'new_defender/run.json').read_text())
        self.assertEqual(saved['attack_contract']['attacker']['revision'],'digest-gemma')
        calls=[body for path,body in Server.calls if path=='/api/chat']
        self.assertEqual(calls[0]['options'],{'temperature':0,'num_predict':1024})
        self.assertEqual(self.run_pipeline('--model','new_defender','--resume'),0)

    def test_full_evaluation_cache_and_final_outcome_selection(self):
        import evaluate_paper
        from types import SimpleNamespace
        self.assertEqual(self.run_pipeline(),0)
        scorer=SimpleNamespace(contract={'test_encoder':'fixed'},ttest=None)
        scorer.score=lambda original,rewrite:dict(sbert=.9,sbert_status='OK',
            sbert_original_truncated=False,sbert_rewrite_truncated=False)
        ledger,rows,saved=evaluate_paper.load_run(self.out,'clef')
        with patch.object(scorer,'score',wraps=scorer.score) as scoring:
            summary=evaluate_paper.evaluate_run(ledger,rows,saved,scorer,False)
            self.assertEqual(scoring.call_count,5)
        self.assertEqual(summary['sbert_status'],'COMPUTED')
        with patch.object(scorer,'score',side_effect=AssertionError('Repeated embedding')):
            evaluate_paper.evaluate_run(ledger,rows,saved,scorer,False)
        pairs=pipeline.read_csv(self.out/'clef/evaluation/similarity_per_rewrite.csv')
        final=[r for r in pairs if r['stage']=='final']
        self.assertEqual([r['source_step'] for r in final],['1','2','2'])
        self.assertEqual([r['outcome'] for r in final],['success','success','failure'])
        cache=next((self.out/'evaluation_cache').glob('*/*.json'))
        changed=json.loads(cache.read_text());changed['values']['sbert']=.1
        cache.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            evaluate_paper.evaluate_run(ledger,rows,saved,scorer,False)

    def test_incomplete_paper_rates_not_reported_as_final(self):
        Server.truncated=True
        self.assertEqual(self.run_pipeline(),2)
        m=self.metrics()
        self.assertIsNone(m['paper_final_asr_pct'])
        self.assertIsNone(m['paper_step2_gain_pct'])
        self.assertIsNone(m['source_modified_asr_pct'])
        self.assertEqual(m['paper_asr_lower_bound_pct'],0)
        self.assertEqual(m['paper_asr_upper_bound_pct'],100)
        first=pipeline.read_csv(self.out/'clef/step1_rewrites.csv')
        self.assertTrue(all(r['generation_status']=='ERROR' and r['rewrite_text']=='' for r in first))


if __name__=='__main__':unittest.main()
