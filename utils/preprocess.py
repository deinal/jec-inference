import numpy as np
import awkward as ak
from functools import partial
from itertools import chain


def _get_variable_names(expr, exclude=['ak', 'np', 'numpy']):
    import ast
    root = ast.parse(expr)
    return sorted({node.id for node in ast.walk(root) if isinstance(
        node, ast.Name) and not node.id.startswith('_')} - set(exclude))

def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return ak.concatenate(arrays, axis=axis)

def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.highlevel.Array):
        none_padded_a = ak.pad_none(a, maxlen, clip=True)
        padded_a = ak.fill_none(none_padded_a, value)
        return ak.values_astype(padded_a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x

def _repeat_pad(a, maxlen, shuffle=False, dtype='float32'):
    x = a.flatten()
    x = np.tile(x, int(np.ceil(len(a) * maxlen / len(x))))
    if shuffle:
        np.random.shuffle(x)
    x = x[:len(a) * maxlen].reshape((len(a), maxlen))
    mask = _pad(ak.zeros_like(a), maxlen, value=1)
    x = _pad(a, maxlen) + mask * x
    return x.astype(dtype)

def _clip(a, a_min, a_max):
    if isinstance(a, np.ndarray):
        return np.clip(a, a_min, a_max)
    else:
        return ak.unflatten(np.clip(a.content, a_min, a_max), a.counts)

def _eval_expr(expr, table):
    tmp = {k: table[k] for k in _get_variable_names(expr)}
    tmp.update(
        {'np': np, 'ak': ak, '_concat': _concat, '_pad': _pad,
         '_repeat_pad': _repeat_pad, '_clip': _clip})
    return eval(expr, tmp)
    
def build_new_variables(table, funcs):
    if funcs is None:
        return
    for k, expr in funcs.items():
        if k in table:
            continue
        table[k] = _eval_expr(expr, table)
        
def finalize_inputs(table, data_config):
    for k, params in data_config.preprocess_params.items():
        if data_config._auto_standardization and params['center'] == 'auto':
            raise ValueError('No valid standardization params for %s' % k)
        if params['center'] is not None:
            table[k] = _clip((table[k] - params['center']) * params['scale'], params['min'], params['max'])
        if params['length'] is not None:
            pad_fn = _repeat_pad if params['pad_mode'] == 'wrap' else partial(_pad, value=params['pad_value'])
            table[k] = pad_fn(table[k], params['length'])
        if isinstance(table[k], ak.highlevel.Array):
            table[k] = ak.to_numpy(table[k])
        # check for NaN
        if np.any(np.isnan(table[k])):
            print('Found NaN in %s, silently converting it to 0.', k)
            table[k] = np.nan_to_num(table[k])
    # stack variables for each input group
    for k, names in data_config.input_dicts.items():
        if len(names) == 1 and data_config.preprocess_params[names[0]]['length'] is None:
            table['_' + k] = table[names[0]]
        else:
            table['_' + k] = np.stack([table[n] for n in names], axis=1)
    # reduce memory usage
    for n in set(chain(*data_config.input_dicts.values())):
        if n not in data_config.label_names and n not in data_config.observer_names:
            del table[n]
