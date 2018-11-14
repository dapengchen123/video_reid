from __future__ import absolute_import
from .ilidsvidsequence import iLIDSVIDSEQUENCE
from .prid2011sequence import PRID2011SEQUENCE


def get_sequence(name, root, *args, **kwargs):
    __factory = {
        'ilidsvidsequence': iLIDSVIDSEQUENCE,
        'prid2011sequence': PRID2011SEQUENCE
    }

    if name not in __factory:
        raise KeyError("Unknown dataset", name)
    return __factory[name](root, *args, **kwargs)