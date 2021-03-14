from __future__ import absolute_import
from .market1501duke10 import market1501



__factory = {
    'market1501duke10': market1501,
}


def names():
    return sorted(__factory.keys())


def create(name, root, trainer, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'market', 'duke'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, trainer, *args, **kwargs)
