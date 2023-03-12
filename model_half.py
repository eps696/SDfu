import os
import argparse
import collections
import pickle
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir", '-d', default='./', help="directory with models")
parser.add_argument("--ext", '-e', default=['ckpt','pt', 'bin'], help="model extensions")
a = parser.parse_args()

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def file_list(path, ext=None):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    if ext is not None: 
        if isinstance(ext, list):
            files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ext]
        elif isinstance(ext, str):
            files = [f for f in files if f.endswith(ext)]
        else:
            print(' Unknown extension/type for file list!')
    return sorted([f for f in files if os.path.isfile(f)])

def float2half(data):
    for k in data:
        if isinstance(data[k], collections.abc.Mapping):
            data[k] = float2half(data[k])
        elif isinstance(data[k], list):
            data[k] = [float2half(x) for x in data[k]]
        else:
            if data[k] is not None and torch.is_tensor(data[k]) and data[k].type() == 'torch.FloatTensor':
                data[k] = data[k].half()
    return data

models = file_list(a.dir, a.ext)

for model_path in models:
    file_out = basename(model_path) + '-half' + os.path.splitext(model_path)[-1]
    model = torch.load(model_path)
    model = float2half(model)
    torch.save(model, file_out)

