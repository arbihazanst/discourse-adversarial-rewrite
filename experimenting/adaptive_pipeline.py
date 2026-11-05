"""Experimental Open WebUI PPLX profile for the shared paper controller."""
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'scripts'))
from adaptive_pipeline import main

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        from env_config import load_project_env
        load_project_env()
        sys.exit(main(['--config',str(HERE/'adaptive_config.json'),*sys.argv[1:]]))
    except (ValueError,OSError,KeyError) as exc:
        print(f'Configuration/data error: {exc}',file=sys.stderr);sys.exit(2)
