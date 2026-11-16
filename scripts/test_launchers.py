"""CLI routing, profile isolation, launchers, and resume-plan regression tests."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import experiment_cli as cli

ROOT=cli.pipeline.ROOT
BASH=str(Path('C:/Program Files/Git/bin/bash.exe')) if Path('C:/Program Files/Git/bin/bash.exe').exists() else shutil.which('bash')


class CliTests(unittest.TestCase):
    def setUp(self):
        self.main=cli.load_profile(cli.MAIN)
        self.exp=cli.load_profile(ROOT/'experimenting/adaptive_config.json')
        for name in ('prepare_gan_before_run','publish_results'):
            mocked=patch.object(cli,name,return_value=0);mocked.start();self.addCleanup(mocked.stop)

    def test_only_four_launchers_remain(self):
        actual={p.relative_to(ROOT).as_posix() for folder in (ROOT,ROOT/'scripts',ROOT/'experimenting')
                for p in folder.iterdir() if p.suffix in ('.bat','.sh')}
        self.assertEqual(actual,{'run_cli.bat','run_cli.sh','experimenting/run_cli.bat','experimenting/run_cli.sh'})

    def test_all_pilot_uses_one_tunnel_run(self):
        for cfg in (self.main,self.exp):
            with patch('tunnel_evaluation.run',return_value=0) as run:
                self.assertEqual(cli.execute_plan(cli.MAIN,cfg,'all','pilot','all',retry=True),0)
                run.assert_called_once_with(cfg,'pilot','all','all',retry=True,dry=False,check=False)

    def test_full_plan_routes_selected_model(self):
        with patch('tunnel_evaluation.run',return_value=0) as run:
            self.assertEqual(cli.main(['--task','gan','--mode','full','--model','d3','--dry-run']),0)
            self.assertEqual(run.call_args.args[1:4],('full','gan','d3'))

    def test_pipeline_error_reaches_caller(self):
        with patch('tunnel_evaluation.run',return_value=2):
            self.assertEqual(cli.execute_plan(cli.MAIN,self.main,'clef','pilot','all'),2)

    def test_profile_and_output_isolation(self):
        plan=cli.build_plan(cli.MAIN,self.main,'clef','pilot','llm')
        self.assertIn('--pilot',plan[0][1])
        self.assertNotIn('pilot_openwebui.py',plan[0][1][1])
        with self.assertRaises(ValueError):
            cli.build_plan(cli.MAIN,self.main,'pplx','pilot','llm')
        bad=copy.deepcopy(self.exp);bad['output_dir']='outputs/wrong'
        with self.assertRaises(ValueError):cli.validate_paths(bad)
        bad=copy.deepcopy(self.main);bad['output_dir']='experimenting/outputs/wrong'
        with self.assertRaises(ValueError):cli.validate_paths(bad)
        bad=copy.deepcopy(self.exp);bad['tool_llm_pilot_output_dir']=bad['tool_pilot_output_dir']
        with self.assertRaises(ValueError):cli.validate_paths(bad)
        with self.assertRaisesRegex(ValueError,'tidak sesuai'):
            cli.main(['--profile','main','--config',str(ROOT/'experimenting/adaptive_config.json'),'--show'])
        with self.assertRaises(ValueError):
            cli.save_profile(ROOT/'decision_models/should_not_exist.json',self.exp)
        self.assertFalse((ROOT/'decision_models/should_not_exist.json').exists())

    def test_noninteractive_dry_run_and_retry_reach_every_stage(self):
        with patch('tunnel_evaluation.run',return_value=0) as run:
            code=cli.main(['--profile','experimenting','--task','all','--dry-run','--retry-errors'])
        self.assertEqual(code,0)
        self.assertEqual(run.call_args.args[0]['profile'],'experimenting')
        self.assertEqual(run.call_args.args[1:8],('pilot','all','all',[],True,False,True))

    def test_menu_back_navigation_and_pilot_execution(self):
        answers=['1','0','1','4','1','0']
        with patch('builtins.input',side_effect=answers),patch('tunnel_evaluation.run',return_value=0) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--profile','experimenting']),0)
        self.assertEqual(run.call_count,1)
        self.assertEqual(run.call_args.args[1:4],('pilot','all','all'))

    def test_saved_outputs_discovery_is_profile_scoped(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'experimenting') as temp:
            cfg=copy.deepcopy(self.exp)
            cfg['tool_pilot_output_dir']=temp
            p=Path(temp)/'pplx';p.mkdir()
            (p/'run.json').write_text('{"defender":{}}')
            (p/'checkpoint.json').write_text('{"state":"complete","selected_count":10}')
            found=cli.saved_runs(cfg)
            self.assertTrue(any(path==p/'run.json' for path,_,_ in found))
            self.assertTrue(all(path.is_relative_to(ROOT/'experimenting') for path,_,_ in found))

    def test_route_overrides_do_not_leak_into_children(self):
        with patch.dict(os.environ,{'PPLX_CHAT_MODEL':'disabled-model','PPLX_MODEL_ID':'disabled-model','OWUI_API_KEY':'private'}):
            env=cli.child_environment(self.exp)
        self.assertNotIn('PPLX_CHAT_MODEL',env);self.assertNotIn('PPLX_MODEL_ID',env)
        self.assertEqual(env['OWUI_API_KEY'],'private')

    @unittest.skipUnless(BASH,'Bash unavailable')
    def test_bash_launchers_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            env=dict(os.environ,PYTHON=sys.executable,EXPERIMENT_PYTHON=sys.executable)
            for relative,profile in [('run_cli.sh','MAIN'),('experimenting/run_cli.sh','EXPERIMENTING')]:
                file=ROOT/relative
                syntax=subprocess.run([BASH,'-n',file.as_posix()],capture_output=True,text=True)
                self.assertEqual(syntax.returncode,0,syntax.stderr)
                result=subprocess.run([BASH,file.as_posix(),'--show'],env=env,cwd=folder,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('DECISION RESEARCH | '+profile,result.stdout)
                self.assertIn('google/gemma-3-27b-it',result.stdout)

    @unittest.skipUnless(os.name=='nt','Windows launchers')
    def test_batch_launchers_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            env=dict(os.environ,PYTHON=sys.executable,EXPERIMENT_PYTHON=sys.executable)
            for relative,profile in [('run_cli.bat','MAIN'),('experimenting/run_cli.bat','EXPERIMENTING')]:
                command='cmd.exe /d /s /c "call "'+str(ROOT/relative)+'" --show"'
                result=subprocess.run(command,env=env,cwd=folder,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn('DECISION RESEARCH | '+profile,result.stdout)


if __name__=='__main__':unittest.main()
