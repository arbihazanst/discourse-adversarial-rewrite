"""Discover a compatible native PPLX POST route from its own OpenAPI document."""
import copy
import json
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import adaptive_pipeline as core

RESOLVED_FIELDS={'native_endpoint','native_request_defaults','native_schema_sha256'}


def base_url(value):
    parsed=urlsplit(value)
    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('PPLX native URL must be an HTTP(S) base URL without credentials, query or fragment')
    if parsed.port is not None and not 1<=parsed.port<=65535:raise ValueError('Invalid PPLX port')
    return value.rstrip('/')


def resolve_schema(spec,value,depth=0):
    if depth>16 or not isinstance(value,dict):raise ValueError('Unsupported or recursive OpenAPI schema')
    if '$ref' in value:
        ref=value['$ref']
        if not isinstance(ref,str) or not ref.startswith('#/'):raise ValueError('Only local OpenAPI references are supported')
        target=spec
        for key in ref[2:].split('/'):target=target[key.replace('~1','/').replace('~0','~')]
        return resolve_schema(spec,target,depth+1)
    if 'allOf' in value:
        merged={'type':'object','properties':{},'required':[]}
        for part in value['allOf']:
            item=resolve_schema(spec,part,depth+1)
            if item.get('type','object')!='object':raise ValueError('Unsupported allOf schema')
            merged['properties'].update(item.get('properties',{}));merged['required']+=item.get('required',[])
        return merged
    return copy.deepcopy(value)


def compatible_request(spec,operation,path_parameters):
    if any(resolve_schema(spec,p).get('required') for p in path_parameters+operation.get('parameters',[])):
        raise ValueError('Required URL/header parameters need a dedicated adapter')
    body=resolve_schema(spec,operation['requestBody'])
    schema=resolve_schema(spec,body['content']['application/json']['schema'])
    if schema.get('type','object')!='object':raise ValueError('Body is not an object')
    props=schema.get('properties',{});expected={'state','instructions','criteria'}
    if not expected<=props.keys():raise ValueError('Expected state, instructions and criteria fields are absent')
    for key in ('state','instructions'):
        if resolve_schema(spec,props[key]).get('type')!='string':raise ValueError('Expected string '+key)
    criteria=resolve_schema(spec,props['criteria'])
    if criteria.get('type')!='object':raise ValueError('Expected criteria mapping')
    additional=criteria.get('additionalProperties')
    if additional not in (None,True):
        if not isinstance(additional,dict) or resolve_schema(spec,additional).get('type')!='string':
            raise ValueError('Criteria must accept string descriptions')
    if additional is False:raise ValueError('Criteria mapping disallows dynamic labels')
    defaults={}
    for name in set(schema.get('required',[]))-expected:
        field=resolve_schema(spec,props[name])
        if 'default' not in field:raise ValueError('Required extra field: '+name)
        defaults[name]=field['default']
    return schema,defaults


def discover(config,timeout=30):
    base=base_url(config['url']);headers={'Accept':'application/json'}
    key=core.credentials(config)
    if key:headers['Authorization']='Bearer '+key
    with urlopen(Request(base+'/openapi.json',headers=headers),timeout=timeout) as response:
        spec=json.load(response)
    candidates=[];available=[];requested=config.get('native_path','')
    for path,item in spec.get('paths',{}).items():
        if not isinstance(item,dict) or 'post' not in item:continue
        available.append(path)
        if requested and requested!=path:continue
        if not path.startswith('/') or path.startswith('//') or any(c in path for c in '{}?#'):
            continue
        try:schema,defaults=compatible_request(spec,item['post'],item.get('parameters',[]))
        except (KeyError,TypeError,ValueError):continue
        candidates.append((path,schema,defaults))
    if len(candidates)!=1:
        raise ValueError('PPLX OpenAPI needs exactly one compatible POST route with state:string, instructions:string, criteria:object. '
            'Compatible routes: '+str([p for p,_,_ in candidates])+'. Available POST routes: '+str(available)+
            '. Set native_path when several routes match, or supply the server schema for a dedicated adapter. No classification was performed.')
    path,schema,defaults=candidates[0]
    config.update(native_endpoint=base+path,native_request_defaults=defaults,
        native_schema_sha256=core.fingerprint(dict(path=path,schema=schema)))
    print('PPLX native API verified from OpenAPI:',config['native_endpoint'],flush=True)
    return config


def reuse_saved_metadata(config,folders,names):
    """Offline only: retain a saved schema identity if all configured fields match."""
    def declared(value):return {k:v for k,v in value.items() if k not in RESOLVED_FIELDS}
    for folder in folders:
        for name in names:
            path=folder/name/'run.json'
            if not path.exists():continue
            saved=json.loads(path.read_text(encoding='utf-8')).get('attack_contract',{}).get('validity',{}).get('evaluator',{})
            if declared(config)==declared(saved):
                config.update({k:saved[k] for k in RESOLVED_FIELDS if k in saved});return


def validate(config,live=False,timeout=30):
    base_url(config['url'])
    if live:
        discover(config,timeout)
        if not config.get('revision'):raise ValueError('Set the actual PPLX native checkpoint/revision in evaluator settings before inference')


def request_payload(config,params):
    return {**config.get('native_request_defaults',{}),**params}
