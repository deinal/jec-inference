from tqdm import tqdm
import numpy as np
import awkward as ak
import argparse
import os
import s3fs

from utils.config import DataConfig
from utils.fileio import read_files
from utils.preprocess import build_new_variables, finalize_inputs

import tritonclient.http as httpclient


def get_data(data_config_file, s3, flist):
    data_config = DataConfig.load(data_config_file)
    table = read_files(flist, data_config.load_branches, s3)
    # define new variables
    build_new_variables(table, data_config.var_funcs)
    # perform input variable standardization, clipping, padding and stacking
    finalize_inputs(table, data_config)
    
    input_data = {key: table[f'_{key}'] for key in data_config.inputs.keys()}
    
    observers = ak.to_pandas({k: table[k] for k in data_config.observers})
    
    return input_data, observers


def infer(triton_client, inputs):
    triton_inputs = []
    for i, key in enumerate(inputs.keys()):
        triton_inputs.append(httpclient.InferInput(name=key, shape=inputs[key].shape, datatype="FP32"))
        triton_inputs[i].set_data_from_numpy(inputs[key])

    triton_outputs = [httpclient.InferRequestedOutput(name='output')]

    results = triton_client.infer(
        model_name='optimal',
        inputs=triton_inputs,
        outputs=triton_outputs
    )
    return results


def get_predictions(triton_client, input_data, data_len, batch_size=100):
    predictions = []

    for i in tqdm(range(0, data_len, batch_size)):
        inputs = {}
        for key in input_data:
            inputs[key] = input_data[key][i:i+batch_size]

        results = infer(triton_client, inputs)

        predictions.append(results.as_numpy(name='output'))

    return np.concatenate(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline Params')
    parser.add_argument('--file', type=str)
    parser.add_argument('--fname', type=str)
    args = parser.parse_args()

    if args.fname:
        fname = args.fname
    else:
        fname = open(args.file, 'r').readline()
    print(fname)

    s3 = s3fs.core.S3FileSystem(anon=True, client_kwargs={'endpoint_url': 'https://s3.cern.ch'})
    flist = [fname]

    pfn_input_data, observers = get_data('https://raw.githubusercontent.com/deinal/weaver/dev/jet-energy-corrections/data/jec_pfn.yaml', s3, flist)
    data_len = observers.shape[0]
    for key in pfn_input_data:
        print(key, pfn_input_data[key].shape)

    pfn_triton_client = httpclient.InferenceServerClient(url='pfn-regressor-ea37f4.dholmber.svc.cluster.local', verbose=False)

    pfn_predictions = get_predictions(pfn_triton_client, pfn_input_data, data_len)
