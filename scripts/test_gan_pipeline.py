"""GAN protocol tests with local defenders and deterministic fake generation."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import adaptive_pipeline as core
import evaluate_paper
import experiment_cli
import gan_pipeline as gan
import test_adaptive_pipeline as fixtures


def contract(config,required=True):
    return {'kind':'gan','seed':config.get('seed',20261104),'resource_files':'TEST_ONLY'}


class FakeEngine:
    calls=[]
    def __init__(self,config):self.contract=contract(config)
    def attack(self,original,sid,victim):
        self.calls.append(sid)
        candidate='first '+original
        labels=victim.get_pred([candidate])
        return candidate if int(labels[0])==0 else None


class GanTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.AdaptiveTests('test_full_protocol_ham_exclusion_two_attempts_and_resume')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.out=self.root/'gan'
        self.fixture.config['gan_output_dir']=str(self.out)
        self.fixture.config['attacker']['revision']='' # GAN never requires Gemma.
        self.fixture.configpath.write_text(json.dumps(self.fixture.config))
        FakeEngine.calls=[]
        for target,value in [('gan_contract',contract),('GanEngine',FakeEngine)]:
            p=patch.object(gan,target,value);p.start();self.addCleanup(p.stop)

    def run_gan(self,*args):
        with contextlib.redirect_stdout(io.StringIO()):
            return gan.main(['--config',str(self.fixture.configpath),'--attempts','1',*args])

    def summary(self):return json.loads((self.out/'clef/metrics.json').read_text())

    def test_full_gan_sst_protocol_and_exports(self):
        self.assertEqual(self.run_gan(),0)
        m=self.summary()
        self.assertEqual((m['eligible'],m['gan_successes'],m['gan_no_adversarial_found']),(3,1,2))
        self.assertAlmostEqual(m['gan_asr_pct'],100/3)
        self.assertEqual(m['gan_modified_asr_pct'],100)
        self.assertEqual(m['classification_timed_requests'],11)
        self.assertEqual(len(fixtures.Server.calls),11)
        self.assertFalse(any(path=='/chat' for path,_ in fixtures.Server.calls))
        rows=core.read_csv(self.out/'clef/gan_rewrites.csv')
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[1]['candidate_text'],'first spam-two')
        self.assertEqual(rows[1]['returned_text'],'')
        self.assertEqual(rows[1]['used_text'],'spam-two')
        self.assertEqual(rows[1]['label_flip'],'0')
        self.assertEqual(len(list((self.out/'clef/rewrites/gan').glob('*.csv'))),3)
        with patch.object(core.api,'request_json',side_effect=AssertionError('Repeated inference')), \
                patch.object(gan.time,'sleep',side_effect=AssertionError('Delay on completed work')):
            self.assertEqual(self.run_gan('--resume','--delay','10'),0)
        self.assertEqual(len(FakeEngine.calls),3)

    def test_reuses_matching_llm_baseline(self):
        self.assertEqual(self.fixture.run_pipeline('--baseline-only'),0)
        fixtures.Server.calls=[]
        self.assertEqual(self.run_gan(),0)
        self.assertEqual(self.summary()['baseline_reused'],5)
        self.assertEqual(len(fixtures.Server.calls),6)
        rows=core.read_csv(self.out/'clef/events.csv')
        self.assertTrue(all(r['source']=='baseline_reuse' for r in rows if r['stage']=='baseline'))

    def test_baseline_mismatch_fails_before_new_requests(self):
        self.fixture.run_pipeline('--baseline-only')
        self.fixture.config['defenders']['clef']['revision']='another-model'
        self.fixture.configpath.write_text(json.dumps(self.fixture.config))
        fixtures.Server.calls=[]
        with self.assertRaisesRegex(ValueError,'baseline source'):
            self.run_gan()
        self.assertEqual(fixtures.Server.calls,[])
        self.assertEqual(self.run_gan('--no-reuse-baseline'),0)

    def test_interrupted_probe_resumes_without_repeating_query(self):
        commit=core.Ledger.commit
        def stop(ledger,event,artifact,selected):
            commit(ledger,event,artifact,selected)
            if event['stage']=='gan_probe':raise KeyboardInterrupt
        with patch.object(core.Ledger,'commit',stop):self.assertEqual(self.run_gan(),130)
        self.assertEqual(len(fixtures.Server.calls),6)
        self.assertEqual(self.run_gan('--resume'),0)
        self.assertEqual(len(fixtures.Server.calls),11)
        self.assertEqual(FakeEngine.calls.count('0'),2) # Regenerated deterministically; query reused.

    def test_interrupted_returned_rewrite_is_not_regenerated(self):
        commit=core.Ledger.commit
        def stop(ledger,event,artifact,selected):
            commit(ledger,event,artifact,selected)
            if event['stage']=='gan_attack':raise KeyboardInterrupt
        with patch.object(core.Ledger,'commit',stop):self.assertEqual(self.run_gan(),130)
        self.assertEqual(self.run_gan('--resume'),0)
        self.assertEqual(len(FakeEngine.calls),3)

    def test_probe_errors_require_retry_and_remain_unresolved(self):
        request=core.api.request_json
        def bad(url,key,payload,*args):
            if payload.get('state')=='first spam-two':return {'unparseable':True},200,1,'test'
            return request(url,key,payload,*args)
        with patch.object(core.api,'request_json',side_effect=bad):self.assertEqual(self.run_gan(),2)
        self.assertIsNone(self.summary()['gan_asr_pct'])
        self.assertEqual(self.summary()['gan_unresolved'],1)
        count=len(fixtures.Server.calls)
        self.assertEqual(self.run_gan('--resume'),2)
        self.assertEqual(len(fixtures.Server.calls),count)
        self.assertEqual(self.run_gan('--retry-errors'),0)

    def test_original_fallback_flip_is_not_a_gan_success(self):
        request=core.api.request_json;seen=0
        def drift(url,key,payload,*args):
            nonlocal seen
            if payload.get('state')=='spam-stubborn':
                seen+=1
                if seen==2:return {'label':'NOT_SPAM'},200,1,'test'
            return request(url,key,payload,*args)
        with patch.object(core.api,'request_json',side_effect=drift):self.assertEqual(self.run_gan(),0)
        self.assertEqual(self.summary()['fallback_label_drift'],1)
        self.assertEqual(self.summary()['gan_successes'],1)

    def test_saved_gan_evaluation_includes_rejected_candidates(self):
        self.run_gan();count=len(fixtures.Server.calls)
        self.assertEqual(evaluate_paper.main(['--run-dir',str(self.out),'--lexical-only']),0)
        rows=core.read_csv(self.out/'clef/evaluation/similarity_per_rewrite.csv')
        candidates=[r for r in rows if r['stage']=='gan_candidate']
        returned=[r for r in rows if r['stage']=='gan_returned']
        self.assertEqual(len(candidates),3);self.assertEqual(len(returned),1)
        self.assertEqual(sum(r['outcome']=='failure' for r in candidates),2)
        self.assertEqual(len(fixtures.Server.calls),count)
        with patch.object(evaluate_paper.metrics,'lexical_scores',side_effect=AssertionError('Repeated scoring')):
            self.assertEqual(evaluate_paper.main(['--run-dir',str(self.out),'--lexical-only']),0)

    def test_dry_run_and_gan_menu_routing(self):
        with patch.object(gan,'GanEngine',side_effect=AssertionError('Loaded weights')):
            self.assertEqual(self.run_gan('--dry-run'),0)
        self.assertFalse(self.out.exists());self.assertEqual(fixtures.Server.calls,[])
        command=experiment_cli.build_command(Path('profile.json'),'clef','pilot',True,'gan')
        self.assertIn(str(experiment_cli.HERE/'gan_pipeline.py'),command)
        self.assertNotIn('--reuse-step1',command)
        self.assertEqual(experiment_cli.output_for({'output_dir':'llm'},'gan'),'outputs/paper3000_gan')


if __name__=='__main__':unittest.main()
