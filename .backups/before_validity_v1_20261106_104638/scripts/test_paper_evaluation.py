"""Metric definition checks and menu routing, independent of inference services."""
import math
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_cli as menu
import paper_evaluation as metrics


class PaperTests(unittest.TestCase):
    def test_lexical_scores_and_content_retention(self):
        original='The project meeting is scheduled for Friday at 10 AM.'
        values=metrics.lexical_scores(original,original)
        for name in ('bleu','rouge_l','jaccard','length_ratio','keyword_retention'):
            self.assertAlmostEqual(values[name],1)
        values=metrics.lexical_scores('alpha beta gamma delta','alpha beta')
        self.assertAlmostEqual(values['jaccard'],.5)
        self.assertAlmostEqual(values['rouge_l'],2/3)
        self.assertAlmostEqual(values['length_ratio'],.5)
        self.assertAlmostEqual(values['keyword_retention'],.5)
        expected=math.exp(-1)*math.exp((math.log(1)+math.log(1)+2*math.log(.1))/4)
        self.assertAlmostEqual(values['bleu'],expected)
        self.assertIsNone(metrics.lexical_scores('','hello')['jaccard'])
        self.assertIsNone(metrics.lexical_scores('a the is','the')['keyword_retention'])
        self.assertEqual(metrics.lexical_scores('red green blue yellow','cat dog fish bird')['bleu'],0)

    def test_outcome_grouping_delta_and_small_samples(self):
        rows=[dict(stage='step1',outcome='success',sbert=.9),
              dict(stage='step1',outcome='success',sbert=.8),
              dict(stage='step1',outcome='failure',sbert=.7),
              dict(stage='step1',outcome='unresolved',sbert=.1)]
        summary=metrics.similarity_summary(rows)
        success=next(r for r in summary if r['stage']=='step1' and r['metric']=='sbert' and r['outcome']=='success')
        self.assertEqual(success['n'],2)
        self.assertAlmostEqual(success['mean'],.85)
        self.assertAlmostEqual(success['std'],math.sqrt(.005))
        self.assertAlmostEqual(success['delta_success_minus_failure'],.15)
        self.assertEqual(success['test_status'],'insufficient_group_size')
        self.assertIsNone(success['p_value'])
        self.assertEqual(metrics.semantic_band(.8),'moderate')
        self.assertEqual(metrics.semantic_band(.75),'moderate')
        self.assertEqual(metrics.semantic_band(.81),'strong')
        self.assertEqual(metrics.semantic_band(.74),'potential_drift')

    def test_lcs_repeated_tokens_and_order(self):
        self.assertAlmostEqual(metrics.lexical_scores('a b a b','b a b a')['rouge_l'],.75)
        self.assertAlmostEqual(metrics.lexical_scores('a b c','c b a')['rouge_l'],1/3)

    def test_welch_test_and_degenerate_groups(self):
        from types import SimpleNamespace
        rows=[dict(stage='step1',outcome=group,sbert=value)
              for group,values in [('success',[.8,.9]),('failure',[.5,.6])] for value in values]
        test=unittest.mock.Mock(return_value=SimpleNamespace(statistic=4.24,pvalue=.05))
        summary=metrics.similarity_summary(rows,test)
        test.assert_called_once_with([.8,.9],[.5,.6],equal_var=False)
        row=next(r for r in summary if r['stage']=='step1' and r['metric']=='sbert' and r['outcome']=='all')
        self.assertEqual(row['test_status'],'OK')
        self.assertEqual(row['p_value'],.05)
        for r in rows:r['sbert']=.8
        row=next(r for r in metrics.similarity_summary(rows,test) if r['stage']=='step1' and r['metric']=='sbert')
        self.assertEqual(row['test_status'],'degenerate_variance')
        self.assertIsNone(row['p_value'])

    def test_menu_commands_keep_selection_and_config(self):
        command=menu.build_command(Path('my profile.json'),'new_defender','pilot',True)
        self.assertIn('new_defender',command)
        self.assertIn('my profile.json',command)
        self.assertIn('--reuse-step1',command)
        self.assertIn('--pilot',command)
        self.assertNotIn('shell=True',command)
        config={'defenders':{'clef':{}}}
        with patch.dict(os.environ,{'CLEF_URL':'old','ATTACKER_URL':'old','OWUI_API_KEY':'test-secret'}):
            env=menu.child_environment(config)
        self.assertNotIn('CLEF_URL',env);self.assertNotIn('ATTACKER_URL',env)
        self.assertEqual(env['OWUI_API_KEY'],'test-secret')

    def test_menu_save_switch_and_dry_run(self):
        with tempfile.TemporaryDirectory(dir=menu.pipeline.ROOT/'scripts') as tmp:
            save=Path(tmp)/'local profile.json'
            # Select d3, save a different output directory, then offline check.
            answers=['5','1','3','4','4','outputs/menu_test','0','2','1','2','0']
            with patch('builtins.input',side_effect=answers),patch.object(menu,'run_command',return_value=0) as run:
                self.assertEqual(menu.main(['--config',str(menu.MAIN),'--save-config',str(save)]),0)
            self.assertEqual(menu.load_profile(save)['output_dir'],'outputs/menu_test')
            command=run.call_args.args[0]
            self.assertIn('d3',command);self.assertIn('--dry-run',command)
            self.assertIn(str(save),command)

    def test_model_key_path_safety(self):
        for name in ('../oops','ALL','all','a/b','generation_cache'):
            with self.assertRaises(ValueError):menu.pipeline.validate_model_key(name)

    @unittest.skipUnless(os.name=='nt','Windows batch launcher')
    def test_windows_menu_launcher(self):
        command=f'call "{menu.pipeline.ROOT / "run_cli.bat"}" --show'
        result=subprocess.run('cmd.exe /d /s /c "'+command+'"',cwd=menu.pipeline.ROOT.parent,
                              env=dict(os.environ,PYTHON=sys.executable),text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('30007/v1/chat/completions',result.stdout)


if __name__=='__main__':unittest.main()
