"""Prepare the official SST GAN resource used by the supplied OpenAttack source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen
import zipfile

import adaptive_pipeline as core

# Official CDN supported by OpenAttack.DataManager.enable_cdn().
URL='https://cdn.data.thunlp.org/TAADToolbox/SGNAE.zip'
FILES=('vocab.json','a.pkl','i.pkl','g.pkl','d.pkl')
DEFAULT_DIR=core.ROOT/'.cache/openattack_gan/AttackAssist.SGAN'


def resource_manifest(folder):
    manifest=json.loads((folder/'resource.json').read_text(encoding='utf-8'))
    for name in FILES:
        path=folder/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'][name]:
            raise ValueError('GAN resource hash mismatch: '+name)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_DIR)
    parser.add_argument('--from-dir',type=Path,help='Import these five files from an existing trusted SGAN resource instead of downloading')
    parser.add_argument('--check-runtime',action='store_true',help='Load the local GAN runtime after resource verification; no generation or API calls')
    parser.add_argument('--verify-only',action='store_true',help='Check existing files only; never download or import')
    args=parser.parse_args(argv);folder=core.rooted(args.output_dir)
    def finish():
        if args.check_runtime:
            from gan_engine import GanEngine
            GanEngine({'resource_dir':str(folder)})
            print('GAN runtime and weights verified; no model API calls.',flush=True)
        return 0
    if (folder/'resource.json').exists():
        resource_manifest(folder);print('GAN weights already prepared and verified:',folder);return finish()
    if args.verify_only:raise ValueError('GAN weights missing. Run prepare_gan.py or choose Run in the CLI to prepare them automatically.')
    folder.parent.mkdir(parents=True,exist_ok=True)
    with core.file_lock(folder.parent/'prepare.lock'):
        if any((folder/name).exists() for name in FILES):
            raise ValueError('Unmanifested GAN files already exist. Import them with --from-dir into a new output directory.')
        with tempfile.TemporaryDirectory(prefix='prepare_gan_',dir=folder.parent) as tmp:
            tmp=Path(tmp);weights=tmp/'weights';weights.mkdir()
            archive_hash=None
            if args.from_dir:
                for name in FILES:shutil.copyfile(args.from_dir/name,weights/name)
                source=str(args.from_dir.resolve())
            else:
                archive=tmp/'SGNAE.zip';size=0;last=0
                print('Downloading the source SST GAN resource from',URL,flush=True)
                with urlopen(URL,timeout=60) as response,archive.open('wb') as dest:
                    while True:
                        block=response.read(1024*1024)
                        if not block:break
                        dest.write(block);size+=len(block)
                        if size-last>=10*1024*1024:print(f'  downloaded {size//(1024*1024)} MiB',flush=True);last=size
                archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
                with zipfile.ZipFile(archive) as z:
                    for name in FILES:
                        entries=[i for i in z.infolist() if not i.is_dir() and Path(i.filename).name==name]
                        if len(entries)!=1:raise ValueError('Expected one resource file: '+name)
                        # Copy only known basenames; never extract arbitrary archive paths.
                        with z.open(entries[0]) as src,(weights/name).open('wb') as dst:shutil.copyfileobj(src,dst)
                source=URL
            vocab=json.loads((weights/'vocab.json').read_text(encoding='utf-8'))
            if not isinstance(vocab,dict) or '<oov>' not in vocab:raise ValueError('Invalid GAN vocabulary')
            manifest=dict(resource='AttackAssist.SGAN',source=source,archive_sha256=archive_hash,
                          files={name:hashlib.sha256((weights/name).read_bytes()).hexdigest() for name in FILES})
            folder.mkdir(parents=True,exist_ok=True)
            for name in FILES:shutil.copyfile(weights/name,folder/name)
            core.atomic_json(folder/'resource.json',manifest)
    print('Prepared GAN weights:',folder)
    return finish()


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,ImportError,RuntimeError,zipfile.BadZipFile) as exc:
        print('GAN setup error:',exc,file=sys.stderr);sys.exit(2)
