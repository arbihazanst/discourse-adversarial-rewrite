import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import adaptive_pipeline as core
import experiment_cli as cli
import results_overview as overview
import prepare_gan
import gan_engine


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((core.ROOT/'decision_models/adaptive_main.json').read_text())
        self.temp=tempfile.TemporaryDirectory(dir=core.ROOT/'scripts');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        mocked=patch.object(core,'ROOT',self.root);mocked.start();self.addCleanup(mocked.stop)

    def run_folder(self,branch):return cli.destination(self.config,'pilot',branch)/'clef'

    def fixture(self):
        for branch in ('baseline','llm'):
            folder=self.run_folder(branch);folder.mkdir(parents=True)
            core.atomic_json(folder/'run.json',{'identities':{'baseline':'b','attack':branch}})
            core.atomic_json(folder/'metrics.json',dict(identities={'baseline':'b','attack':branch},selected_rows=10,eligible=4,observed_valid_flips=0,invalid=1,still_spam=3))
            core.atomic_json(folder/'checkpoint.json',{'state':'complete'})
            core.atomic_csv(folder/'sample_results.csv',[dict(sample_id='A',true_label='SPAM',baseline_label='SPAM',state='invalid_step1')],['sample_id','true_label','baseline_label','state'])
        core.atomic_csv(self.run_folder('llm')/'step1_rewrites.csv',[
            dict(defender='clef',sample_id='A',step=1,validation_status='INVALID',validation_reason='purpose',original_text='original',rewrite_text='candidate')],
            ['defender','sample_id','step','validation_status','validation_reason','original_text','rewrite_text'])
        core.atomic_csv(self.run_folder('llm')/'events.csv',[
            dict(sample_id='A',stage='validate1',status='ERROR',error='old error'),
            dict(sample_id='A',stage='validate1',status='OK',error=''),
            dict(sample_id='B',stage='validate1',status='ERROR',error='current error')],['sample_id','stage','status','error'])

    def test_flat_report_keeps_sources_and_marks_missing_gan(self):
        self.fixture()
        originals={p:p.read_bytes() for base in (self.run_folder('baseline'),self.run_folder('llm')) for p in base.glob('*')}
        with patch.object(core.api,'request_json',side_effect=AssertionError('Network during report')):
            report=overview.refresh(self.config,'pilot')
        self.assertIn('BELUM DIJALANKAN',report.read_text())
        self.assertEqual(len(core.read_csv(report.parent/'gemma_step1.csv')),1)
        self.assertEqual(core.read_csv(report.parent/'gan_step1.csv'),[])
        self.assertEqual(core.read_csv(report.parent/'errors.csv')[0]['error'],'current error')
        self.assertEqual(len(core.read_csv(report.parent/'errors.csv')),1)
        self.assertEqual({p:p.read_bytes() for p in originals},originals)
        self.assertTrue((report.parent.parent/'BUKA_HASIL.md').exists())
        # Regenerating cannot retain stale copied rows after a source export changes.
        core.atomic_csv(self.run_folder('llm')/'step1_rewrites.csv',[],['sample_id'])
        overview.refresh(self.config,'pilot')
        self.assertEqual(core.read_csv(report.parent/'gemma_step1.csv'),[])

    def test_identity_mismatch_is_not_silently_summarized(self):
        self.fixture();core.atomic_json(self.run_folder('llm')/'metrics.json',{'identities':{'attack':'wrong'}})
        with self.assertRaisesRegex(ValueError,'identity mismatch'):overview.refresh(self.config,'pilot')

    def test_preflight_failure_stops_before_baseline(self):
        with patch.object(cli,'run_command',return_value=2) as run,patch.object(cli,'publish_results') as report,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.execute_plan(Path('config.json'),self.config,'clef','pilot','all'),2)
        self.assertEqual(run.call_count,1)
        self.assertEqual(Path(run.call_args.args[0][1]).name,'prepare_gan.py')
        self.assertIn('--check-runtime',run.call_args.args[0]);report.assert_called_once()

    def test_successful_preflight_then_three_tasks(self):
        with patch.object(cli,'run_command',return_value=0) as run,patch.object(cli,'publish_results') as report,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.execute_plan(Path('config.json'),self.config,'clef','pilot','all'),0)
        self.assertEqual(run.call_count,4);self.assertEqual(report.call_count,3)
        self.assertEqual(Path(run.call_args_list[0].args[0][1]).name,'prepare_gan.py')
        self.assertIn('--baseline-only',run.call_args_list[1].args[0])

    def test_dry_run_skips_setup_and_check_only_never_downloads(self):
        with patch.object(cli,'run_command',return_value=0) as run,patch.object(cli,'publish_results') as report,contextlib.redirect_stdout(io.StringIO()):
            cli.execute_plan(Path('config.json'),self.config,'clef','pilot','all',dry=True)
            self.assertEqual(run.call_count,3);report.assert_not_called()
            run.reset_mock()
            cli.execute_plan(Path('config.json'),self.config,'clef','pilot','all',check=True)
            self.assertIn('--verify-only',run.call_args_list[0].args[0]);report.assert_not_called()

    def test_prepare_verify_only_missing_never_downloads(self):
        with patch.object(prepare_gan,'urlopen',side_effect=AssertionError('Download during verify')):
            with self.assertRaisesRegex(ValueError,'missing'):
                prepare_gan.main(['--output-dir',str(self.root/'missing'),'--verify-only','--check-runtime'])
        self.assertFalse((self.root/'missing').exists())

    def test_prepared_resource_loads_runtime_without_download(self):
        folder=self.root/'weights';folder.mkdir();(folder/'resource.json').write_text('{}')
        with patch.object(prepare_gan,'resource_manifest'),patch.object(prepare_gan,'urlopen',side_effect=AssertionError('Unnecessary download')),patch.object(gan_engine,'GanEngine') as engine,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare_gan.main(['--output-dir',str(folder),'--verify-only','--check-runtime']),0)
        engine.assert_called_once_with({'resource_dir':str(folder)})


if __name__=='__main__':unittest.main()
