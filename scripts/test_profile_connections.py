"""Profile routing, revision fallback, and tunnel lifecycle regression tests."""
import copy
import json
import unittest
from unittest.mock import Mock, patch

import adaptive_pipeline as core
import check_model_tunnel as connection
import experiment_cli as cli


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.main=json.loads((core.ROOT/'decision_models/adaptive_main.json').read_text())
        self.exp=json.loads((core.ROOT/'experimenting/adaptive_config.json').read_text())

    def test_main_never_starts_ssh(self):
        with patch.object(connection.subprocess,'Popen') as start, patch.object(connection,'listening') as probe:
            with connection.tunnel(self.main) as proc:self.assertIsNone(proc)
            start.assert_not_called()
            probe.assert_not_called()

    def test_missing_forwards_use_server_ports_and_existing_ports_are_reused(self):
        proc=Mock()
        proc.poll.return_value=None
        with patch.object(connection,'listening',side_effect=[True,False,False,True,True,True]),patch.object(connection.subprocess,'Popen',return_value=proc) as start:
            with connection.tunnel(self.exp):
                command=start.call_args.args[0]
                self.assertEqual([command[i+1] for i,s in enumerate(command) if s=='-L'],
                                 ['127.0.0.1:11439:127.0.0.1:11439','127.0.0.1:11440:127.0.0.1:11440'])
                proc.terminate.assert_not_called()
            proc.terminate.assert_called_once()
        with patch.object(connection,'listening',return_value=True),patch.object(connection.subprocess,'Popen') as start:
            with connection.tunnel(self.exp):pass
            start.assert_not_called()

    def test_empty_revision_has_stable_identity_offline_and_live(self):
        declared=copy.deepcopy(self.main['defenders']['clef'])
        offline,live=copy.deepcopy(declared),copy.deepcopy(declared)
        core.validate_endpoint(offline,live=False)
        core.validate_endpoint(live,live=True)
        self.assertEqual(offline,live)
        self.assertEqual(live['revision_status'],'UNVERIFIED_WEIGHTS')
        first=core.fingerprint(live)
        core.validate_endpoint(live,live=True)
        self.assertEqual(first,core.fingerprint(live))
        live['revision']='sha256:actual-weight-checksum'
        self.assertNotEqual(first,core.fingerprint(live))

    def test_all_uses_same_direct_pipeline_for_both_profiles(self):
        for config in (self.main,self.exp):
            plan=cli.build_plan(cli.MAIN,config,'all','pilot','all')
            self.assertEqual([p[0] for p in plan],['baseline','llm','gan'])
            for _,command,_ in plan:
                self.assertIn('--pilot',command)
                self.assertIn('--resume',command)
                self.assertNotIn('pilot_openwebui.py',str(command))
        body=core.defender_payload(self.main['defenders']['clef'],'test email')
        self.assertEqual(body['model'],'clef-flash-q4')
        self.assertEqual(body['state'],'test email')

    def test_main_check_uses_local_server_endpoints(self):
        def fake(url,key,body,timeout,attempts):
            response=({'choices':[{'message':{'content':'Test reply'}}]} if 'messages' in body
                      else {'answers':{'department':{'choice':'billing'}}})
            return response,200,1,'test'
        with patch.object(connection,'request_json',side_effect=fake) as call,patch.object(connection.subprocess,'Popen') as start:
            self.assertEqual(connection.check(config=self.main),0)
            start.assert_not_called()
            self.assertEqual([c.args[0] for c in call.call_args_list],
                             [r['url'] for r in connection.routes(self.main).values()])
