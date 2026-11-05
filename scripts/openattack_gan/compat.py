"""Minimal interfaces for OpenAttack.GANAttacker.attack(); no package-wide imports."""
class ClassificationAttacker:
    pass


class Classifier:
    pass


class ClassifierGoal:
    def __init__(self, target, targeted):
        self.target, self.targeted = target, targeted

    def check(self, text, prediction):
        return prediction == self.target if self.targeted else prediction != self.target


class Tag:
    def __init__(self, name, category):
        self.name, self.category = name, category


TAG_English = Tag('english', 'lang')


class DataManager:
    resource_dir = None

    @classmethod
    def load(cls, name):
        if name != 'AttackAssist.SGAN' or cls.resource_dir is None:
            raise ValueError('Only the source default SST GAN is configured; run prepare_gan.py first')
        from .sgan import LOAD
        return LOAD(str(cls.resource_dir))


def make_zip_downloader(url):
    def download(*args, **kwargs):
        raise RuntimeError('Use prepare_gan.py for explicit, validated resource preparation')
    return download
