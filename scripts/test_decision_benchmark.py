#!/usr/bin/env python3
"""Offline tests, including a fake local Open WebUI server (never contacts a real API)."""
import csv
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from run_decision_benchmark import choose_decision, extract, load_dataset

HERE=Path(__file__).parent

class FakeOWUI(BaseHTTPRequestHandler):
    calls=0
    def log_message(self, fmt,*args):
        pass
    def do_POST(self):
        assert self.path == '/api/chat/completions'
        data=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
        self.__class__.calls+=1
        if data['model'] != 'decision_spam.clef':
            self.send_response(404);self.end_headers();return
        text=data['messages'][0]['content']
        ans='NOT_SPAM' if 'meeting' in text or 'rewritten' in text else 'SPAM'
        result={'choices':[{'message':{'content':json.dumps({'model':'clef','label':ans,'probabilities':{'SPAM':0.9 if ans=='SPAM' else 0.1, 'NOT_SPAM':0.1 if ans=='SPAM' else 0.9}})}}]}
        body=json.dumps(result).encode('utf8')
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)

class Tests(unittest.TestCase):
    def test_pipe(self):
        resp={'choices':[{'message':{'content':'{"label":"SPAM","probabilities":{"SPAM":0.6,"NOT_SPAM":0.4}}'}}]}
        self.assertEqual(extract(resp,'pipe'),('SPAM','0.6','0.4'))

    def test_native_nested(self):
        self.assertEqual(choose_decision({'answers':{'spam':{'choice':'NOT_SPAM', 'probabilities':{'SPAM':0.1,'NOT_SPAM':0.9}}}}),('NOT_SPAM','0.1','0.9'))

    def test_no_free_form_label(self):
        with self.assertRaises(ValueError):
            extract({'choices':[{'message':{'content':'I think it is SPAM, confidence 0.99'}}]},'pipe')

    def test_tool_source_only(self):
        with self.assertRaises(ValueError):
            extract({'choices':[{'message':{'content':'{"label":"SPAM"}'}}]},'tool')
        self.assertEqual(extract({'sources':[{'type':'tool', 'id':'pplx', 'content':{'result':{'label':'SPAM'}}}]},'tool','pplx')[0],'SPAM')

    def test_real_dataset_shape(self):
        p=HERE.parent/'dataset/benchmark_inputs_clean1484_plus_rewrite742.csv'
        a=load_dataset(p,{'clean','frozen_primary_step1'})
        self.assertEqual(len(a),2226)
        self.assertEqual(sum(r['kind']=='frozen_primary_step1' for r in a),742)

    def test_end_to_end_resume_and_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            d=root/'inputs.csv';out=root/'results.csv';stats=root/'summary.csv'
            with d.open('w',encoding='utf-8-sig',newline='') as fh:
                w=csv.DictWriter(fh,fieldnames=['input_id','sample_id','kind','true_label','email_text','source_attack_model'])
                w.writeheader()
                w.writerows([
                    {'input_id':'1:clean','sample_id':'1','kind':'clean','true_label':'SPAM','email_text':'win prize','source_attack_model':''},
                    {'input_id':'2:clean','sample_id':'2','kind':'clean','true_label':'NOT_SPAM','email_text':'meeting agenda','source_attack_model':''},
                    {'input_id':'1:frozen','sample_id':'1','kind':'frozen_primary_step1','true_label':'SPAM','email_text':'rewritten message','source_attack_model':'some-LLM'},
                ])
            FakeOWUI.calls=0
            httpd=HTTPServer(('127.0.0.1',0),FakeOWUI)
            t=threading.Thread(target=httpd.serve_forever,daemon=True);t.start()
            try:
                cmd=[sys.executable,str(HERE/'run_decision_benchmark.py'),'--dataset',str(d), '--model','clef','--base-url',f'http://127.0.0.1:{httpd.server_port}', '--out',str(out), '--attempts','1']
                env={**os.environ,'OWUI_API_KEY':'test-local'}
                a=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                self.assertEqual(a.returncode,0,a.stderr)
                self.assertEqual(FakeOWUI.calls,3)
                b=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                self.assertEqual(b.returncode,0,b.stderr)
                self.assertEqual(FakeOWUI.calls,3)
                with out.open(newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),3)
                c=subprocess.run([sys.executable,str(HERE/'summarize_decision_results.py'),'--dataset',str(d),'--predictions',str(out),'--out',str(stats)],capture_output=True,text=True)
                self.assertEqual(c.returncode,0,c.stderr)
                with stats.open(newline='') as f:result=next(csv.DictReader(f))
                self.assertEqual(result['clean_accuracy'],'100.00%')
                self.assertEqual(result['observed_flips'],'1')
                self.assertEqual(result['label_flip_rate'],'100.00%')
            finally:
                httpd.shutdown();httpd.server_close();t.join(timeout=4)

if __name__=='__main__':unittest.main(verbosity=2)
