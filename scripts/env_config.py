"""Read a project .env as literal configuration, without executing shell code."""
import os
from pathlib import Path
import re

PROJECT_ENV=Path(__file__).resolve().parent.parent/'.env'


def parse_env(text):
    values={}
    for number,line in enumerate(text.lstrip('\ufeff').splitlines(),1):
        line=line.strip()
        if not line or line.startswith('#'):continue
        match=re.fullmatch(r'(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)',line)
        if not match:raise ValueError(f'Invalid .env assignment on line {number}; use NAME=value')
        name,value=match.groups()
        if value.startswith(('"',"'")):
            quoted=re.fullmatch(r'''(['"])(.*?)\1\s*(?:#.*)?''',value)
            if not quoted:raise ValueError(f'Invalid .env quotes on line {number}')
            value=quoted.group(2)
        else:value=re.split(r'\s+#',value,maxsplit=1)[0].rstrip()
        values[name]=value
    return values


def load_project_env(path=None,allowed_keys=None):
    """Fill unset/empty process values; nonempty terminal values take priority.

    Returns only the names loaded. No values, shell expansion, or output.
    Missing .env is allowed so ordinary exported credentials still work.
    """
    target=Path(os.environ.get('DECISION_ENV_FILE',str(PROJECT_ENV))) if path is None else Path(path)
    if allowed_keys is None and 'DECISION_ENV_KEYS_ONLY' in os.environ:
        allowed_keys=set(os.environ['DECISION_ENV_KEYS_ONLY'].split(','))
    if not target.exists():return set()
    values=parse_env(target.read_text(encoding='utf-8-sig'))
    loaded=set()
    for name,value in values.items():
        if allowed_keys is not None and name not in allowed_keys:continue
        if not os.environ.get(name,'').strip():
            os.environ[name]=value;loaded.add(name)
    return loaded
