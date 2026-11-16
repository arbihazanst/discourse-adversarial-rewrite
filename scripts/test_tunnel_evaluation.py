"""Regression tests for the tunnel experiment. No SSH or inference server required."""
import csv
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import tunnel_evaluation as ev


class TunnelEvaluationTests(TestCase):
    def test_all_gates_candidates_and_resumes(self):
        cfg = json.loads((ev.core.ROOT/'decision_models/adaptive_main.json').read_text())
        rows = [dict(sample_id='a', true_label='SPAM', email_text='Fixture A'),
                dict(sample_id='b', true_label='SPAM', email_text='Fixture B'),
                dict(sample_id='c', true_label='NOT_SPAM', email_text='Fixture C')]
        calls = []
        def reply(url, key, payload, timeout, attempts):
            calls.append((url, payload))
            model = payload['model']
            if 'messages' in payload:
                response = {'choices':[{'message':{'content':'Peringatan: email ini ditandai sebagai spam.'}}]}
            else:
                q = next(iter(payload['questions']))
                if q == 'spam':
                    value = 'NOT_SPAM' if payload['state'] == 'Fixture C' else 'SPAM'
                else:
                    value = 'FAIL' if 'Peringatan:' in payload['state'] else 'PASS'
                response = {'answers':{q:{'choice':value}}}
            return {'model':model, **response}, None, 10, None
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root/'gan.csv'
            fields = ['sample_id','step','original_sha256','parent_sha256','candidate','source_csv','source_record']
            with archive.open('w',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f,fieldnames=fields)
                w.writeheader()
                w.writerow(dict(sample_id='a',step=1,original_sha256=ev.sha('Fixture A'),parent_sha256='',
                                candidate='Fixed text',source_csv='fixture.csv',source_record=1))
            cfg['gan_archive'] = str(archive)
            with mock.patch.object(ev.core.api,'load_dataset',return_value=rows), mock.patch.object(
                    ev.core.api,'request_json',side_effect=reply), mock.patch.object(ev,'service_connection',return_value=nullcontext()):
                for attempt in range(2):
                    self.assertEqual(ev.run(cfg,'full','all',output=root/'run'),0)
                    if attempt == 0: first_count = len(calls)
                    else: self.assertEqual(len(calls),first_count)
            folder = next((root/'run').iterdir())
            report = json.loads((folder/'ringkasan.json').read_text())
            self.assertEqual(report['steps']['gan_1_clef']['status'],{'CLASSIFIED':1,'NO_ARCHIVE':1})
            self.assertEqual(report['steps']['gan_2_d3']['status'],{'NO_ARCHIVE':1})
            self.assertEqual(report['steps']['gemma_1_d3']['status'],{'INVALID':2})
            self.assertFalse(ev.core.read_csv(folder/'gemma_step2.csv'))
            self.assertEqual(set(url for url,_ in calls),{r['url'] for r in cfg['service_pilot'].values()})
            self.assertEqual(sum(1 for _,p in calls if 'messages' in p),2)
            self.assertEqual(sum(1 for _,p in calls if p.get('state') == 'Fixed text'),2)

    def test_reject_unmatched_original(self):
        with mock.patch.object(ev.core,'read_csv',return_value=[dict(sample_id='a',step=1,candidate='fixed',original_sha256='other')]):
            with self.assertRaisesRegex(ValueError,'tidak cocok'):
                ev.load_archive(Path('unused'),{'a':'Original'})
