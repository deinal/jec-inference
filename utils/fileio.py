import numpy as np
import awkward as ak
import uproot
from collections import defaultdict


def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return ak.concatenate(arrays, axis=axis)
    
def _read_root(filepath, branches, treename=None):
    with uproot.open(filepath) as f:
        if treename is None:
            treenames = set([k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree'])
        if len(treenames) == 1:
            treename = treenames.pop()
        else:
            raise RuntimeError('Need to specify `treename` as more than one trees are found')
        tree = f[treename]
        outputs = tree.arrays(branches)
    return outputs

def read_files(filelist, branches, s3):
    branches = list(branches)
    table = defaultdict(list)
    for filepath in filelist:
        filepath = s3.open(filepath)
        a = _read_root(filepath, branches)
        filepath.close()
        for name in branches:
            table[name].append(ak.values_astype(a[name], 'float32'))
    table = {name:_concat(arrs) for name, arrs in table.items()}
    return table
