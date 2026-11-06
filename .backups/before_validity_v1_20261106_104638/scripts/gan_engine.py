"""Pinned source SST GAN runtime and reproducible per-input generation."""
import hashlib
import importlib.metadata
from pathlib import Path
import random

import adaptive_pipeline as core
from prepare_gan import resource_manifest, DEFAULT_DIR


def gan_contract(config, required=True):
    folder=core.rooted(config.get('resource_dir',DEFAULT_DIR))
    try:resource=resource_manifest(folder)
    except FileNotFoundError:
        if required:raise ValueError('GAN weights are missing. Run python scripts/prepare_gan.py first.')
        resource={'files':'NOT_PREPARED'}
    source=Path(__file__).resolve().parent/'openattack_gan'
    return dict(kind='gan',algorithm='OpenAttack.GANAttacker(sst)',seed=int(config.get('seed',20261104)),
        resource_files=resource['files'],source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
        runtime={name:importlib.metadata.version(name) if _installed(name) else 'MISSING' for name in ('torch','numpy')},
        torch_threads=1,adapter_version=1,query_policy='SST candidate probe then explicit post-attack verification',
        max_source_tokens=100,baseline_label_mapping={'SPAM':1,'NOT_SPAM':0})


def _installed(name):
    try:importlib.metadata.version(name);return True
    except importlib.metadata.PackageNotFoundError:return False


class GanEngine:
    def __init__(self, config):
        self.contract=gan_contract(config)
        try:
            import torch
            import numpy as np
            from openattack_gan.attacker import GANAttacker
            from openattack_gan.compat import DataManager, ClassifierGoal
        except ImportError as exc:
            raise ValueError('Install scripts/requirements-gan.txt before running GAN') from exc
        self.torch,self.np,self.goal_type=torch,np,ClassifierGoal
        torch.set_num_threads(1)
        DataManager.resource_dir=core.rooted(config.get('resource_dir',DEFAULT_DIR))
        self.attacker=GANAttacker(gan_dataset='sst')

    def attack(self, original, sample_id, victim):
        # Stable replay after interruption; independent of row ordering or defender.
        seed=int(hashlib.sha256((str(self.contract['seed'])+'\0'+sample_id+'\0'+original).encode()).hexdigest()[:8],16)
        random_state=random.getstate();np_state=self.np.random.get_state()
        try:
            with self.torch.random.fork_rng(devices=[]),self.torch.no_grad():
                random.seed(seed);self.np.random.seed(seed);self.torch.manual_seed(seed)
                return self.attacker.attack(victim,original,self.goal_type(1,targeted=False))
        finally:
            random.setstate(random_state);self.np.random.set_state(np_state)
