#!/usr/bin/env python3
"""List Open WebUI model identifiers to verify the Clef/d3 Pipe registrations."""
import json
import os
import sys
from urllib.request import Request,urlopen
from urllib.error import HTTPError, URLError
url=os.environ.get('OWUI_BASE_URL','https://llms.si-lab.org').rstrip('/')+'/api/models'
key=os.environ.get('OWUI_API_KEY','').strip()
if not key:
    raise SystemExit('Set OWUI_API_KEY before running this script')
req=Request(url,headers={'Authorization':'Bearer '+key,'Accept':'application/json'})
try:
    with urlopen(req,timeout=30) as r: data=json.load(r)
except (HTTPError,URLError,TimeoutError) as e:
    raise SystemExit('Cannot read Open WebUI models: '+str(e))
rows = data.get('data',data.get('models',[])) if isinstance(data,dict) else data
for m in rows:
    if isinstance(m,dict):
        print(m.get('id',''), '|',m.get('name',''))
