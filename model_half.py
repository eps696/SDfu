import os
import argparse
import collections
import pickle
from shutil import move
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", '-i', default='./', help="file or directory with models")
parser.add_argument("--ext", '-e', default=['ckpt','pt', 'bin','safetensors'], help="model extensions")
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
            data[k] = [float2half(x) for x in data[k] if not isinstance(x, int)]
        else:
            if data[k] is not None and torch.is_tensor(data[k]) and data[k].type() in ['torch.FloatTensor', 'torch.cuda.FloatTensor']:
                data[k] = data[k].half()
    return data

models = [a.input] if os.path.isfile(a.input) else file_list(a.input, a.ext)

if any(['safetensors' in f for f in models]):
    import safetensors.torch as safe

for model_path in models:
    issafe = '.safetensors' in model_path.lower()
    model = safe.load_file(model_path) if issafe else torch.load(model_path)

    model = float2half(model)

    file_bak = basename(model_path) + '-full' + os.path.splitext(model_path)[-1]
    move(model_path, file_bak)
    if issafe:
        safe.save_file(model, model_path)
    else:
        torch.save(model, model_path)

