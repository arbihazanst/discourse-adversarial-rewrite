import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import native_pplx as native
import rewrite_validity as validity
import experiment_cli as cli
from test_rewrite_validity import ValidityTests


def specification(paths=('/classify',)):
    schema={'type':'object','required':['state','instructions','criteria'],'properties':{
        'state':{'type':'string'},'instructions':{'type':'string'},'criteria':{'type':'object','additionalProperties':{'type':'string'}}}}
    operation={'requestBody':{'content':{'application/json':{'schema':{'$ref':'#/components/schemas/Input'}}}}}
    return {'openapi':'3.1.0','paths':{path:{'post':copy.deepcopy(operation)} for path in paths},'components':{'schemas':{'Input':schema}}}


class NativeContractTests(unittest.TestCase):
    def config(self):return dict(transport='native',url='http://127.0.0.1:30002',native_path='',revision='real-test-revision',key_env='')

    def discover(self,config,spec):
        with patch.object(native,'urlopen',return_value=io.BytesIO(json.dumps(spec).encode())) as get,contextlib.redirect_stdout(io.StringIO()):
            result=native.discover(config)
        self.assertEqual(get.call_args.args[0].full_url,config['url']+'/openapi.json')
        return result

    def test_discovers_exact_route_and_preserves_rubric_payload(self):
        config=self.discover(self.config(),specification())
        self.assertEqual(config['native_endpoint'],'http://127.0.0.1:30002/classify')
        params=validity.parameters('original','rewrite','meaning')
        self.assertEqual(validity.payload(config,params),params)
        self.assertEqual(validity.parse_response(config,{'result':{'choice':'PASS'}},params),'PASS')

    def test_ambiguous_unsupported_and_required_query_are_rejected(self):
        for spec in (specification(('/one','/two')),specification()):
            if len(spec['paths'])==1:spec['components']['schemas']['Input']['properties']['criteria']={'type':'array'}
            with self.assertRaisesRegex(ValueError,'exactly one'):self.discover(self.config(),spec)
        spec=specification();spec['paths']['/classify']['post']['parameters']=[{'in':'query','name':'extra','required':True}]
        with self.assertRaises(ValueError):self.discover(self.config(),spec)

    def test_manual_path_selects_only_verified_operation(self):
        config=self.config();config['native_path']='/two'
        self.assertTrue(self.discover(config,specification(('/one','/two')))['native_endpoint'].endswith('/two'))
        config['native_path']='/guessed'
        with self.assertRaises(ValueError):self.discover(config,specification())

    def test_rejects_chat_only_or_conflicting_native_response(self):
        for response in ({'choices':[{'message':{'content':'PASS'}}]}, {'choice':'SPAM'}, {'result':[{'choice':'PASS'},{'choice':'FAIL'}]}):
            with self.assertRaises(ValueError):validity.parse_response(self.config(),response,{})

    def test_required_extra_fields_need_defaults(self):
        spec=specification();schema=spec['components']['schemas']['Input']
        schema['required'].append('batch_size');schema['properties']['batch_size']={'type':'integer'}
        with self.assertRaises(ValueError):self.discover(self.config(),spec)
        schema['properties']['batch_size']['default']=1
        config=self.discover(self.config(),spec)
        self.assertEqual(validity.payload(config,{'state':'email'})['batch_size'],1)

    def test_port_edit_keeps_protocol_path_model_and_credentials(self):
        original=dict(transport='openai',url='http://127.0.0.1:30007/v1/chat/completions',model='gemma',key_env='KEY',revision='r')
        with patch('builtins.input',side_effect=['gpu-host','31007']),contextlib.redirect_stdout(io.StringIO()):changed=cli.edit_host_port(original)
        self.assertEqual(changed['url'],'http://gpu-host:31007/v1/chat/completions')
        self.assertEqual({k:v for k,v in original.items() if k!='url'},{k:v for k,v in changed.items() if k!='url'})
        self.assertEqual(original['url'],'http://127.0.0.1:30007/v1/chat/completions')

    def test_invalid_ports_and_hosts_are_rejected_and_ipv6_supported(self):
        for answer in ('0','65536','-1','abc'):
            with patch('builtins.input',side_effect=['localhost',answer]),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):cli.edit_host_port(self.config())
        with patch('builtins.input',side_effect=['http://host','30002']),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):cli.edit_host_port(self.config())
        with patch('builtins.input',side_effect=['::1','31002']),contextlib.redirect_stdout(io.StringIO()):changed=cli.edit_host_port(self.config())
        self.assertEqual(changed['url'],'http://[::1]:31002')


class NativeFlowTests(ValidityTests):
    # Inherit the existing mock scenario; run only an additional native test by
    # keeping the inherited fixtures but not rediscovering their test methods.
    def test_native_main_gate_resume_and_offline_identity(self):
        self.config.update(profile='main',output_dir=str(self.root/'outputs/native'))
        self.config['defenders']={'clef':dict(transport='direct',url='http://mock/clef',model='clef',revision='test',key_env='')}
        self.config['active_models']=['clef']
        self.config['rewrite_evaluator']=dict(transport='native',url='http://mock:30002',model='pplx',revision='test',key_env='',role='pplx_validity',rubric_version=validity.VERSION,native_path='')
        calls=[]
        def request(url,key,body,timeout,attempts):
            calls.append((url,body))
            if url.endswith('/clef'):return {'label':'SPAM'},200,1,''
            if url.endswith('/classify'):return {'choice':'FAIL'},200,1,''
            return self.request(url,key,body,timeout,attempts)
        import validity_pipeline as pipeline
        with patch.object(native,'urlopen',side_effect=lambda *a,**kw:io.BytesIO(json.dumps(specification()).encode())),patch.object(pipeline.core.api,'request_json',side_effect=request),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(['--model','clef'],self.config),0)
            before=len(calls)
            self.assertEqual(pipeline.main(['--model','clef'],self.config),0)
            self.assertEqual(len(calls),before)
            with patch.object(native,'urlopen',side_effect=AssertionError('Network in dry-run')):
                self.assertEqual(pipeline.main(['--model','clef','--dry-run'],self.config),0)
        self.assertEqual(sum(url.endswith('/clef') for url,_ in calls),9)
        self.assertTrue(any(url=='http://mock:30002/classify' for url,_ in calls))
