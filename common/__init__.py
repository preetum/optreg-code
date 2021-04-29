import os
import pickle
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm.auto import tqdm, trange
import pylab as plt
import pickle
import subprocess
import os.path as path
from os.path import join as pjoin

# import gcsfs
# def gopen(gsname, mode='rb'):
#     fs = gcsfs.GCSFileSystem()
#     return fs.open(gsname, mode)
# def glob(gspath):
#     fs = gcsfs.GCSFileSystem()
#     return fs.glob(gspath)

import tensorflow as tf
glob=tf.io.gfile.glob

def gopen(gsname, mode='rb'):
    import tensorflow as tf
    return tf.io.gfile.GFile(gsname, mode)

def gsave(x, gsname):
    with gopen(gsname, 'wb') as f:
        pickle.dump(x, f)

def gload(gsname):
    with gopen(gsname, 'rb') as f:
        x = pickle.load(f)
    return x

def download_dir(gpath, localroot='~/tmp/data', no_clobber=True):
    ''' Downloads GCS dir into localdir (if not exists), and returns the local dir path.'''
    import subprocess
    import pickle
    localroot = path.expanduser(localroot)

    nc = '-n' if no_clobber else ''
    subprocess.call(f'mkdir -p {localroot}', shell=True)
    subprocess.call(f'gsutil -m cp {nc} -r {gpath} {localroot}', shell=True)
    localdir = pjoin(localroot, path.basename(gpath))
    return localdir

def dload(gpath, localdir='~/tmp/data'):
    ''' Downloads from GCS and unpickles into memory'''
    import subprocess
    import pickle

    localdir = os.path.expanduser(localdir)
    local_fname = f'{localdir}/file'
    subprocess.call(f'mkdir -p {localdir}', shell=True)
    subprocess.call(f'gsutil -m cp {gpath} {local_fname}', shell=True)
    with open(os.path.expanduser(local_fname), 'rb') as f:
        obj = pickle.load(f)
    return obj

def call(cmd):
    subprocess.call(cmd, shell=True)

def mse_loss(output, y):
    y_true = F.one_hot(y, 10).float()
    return (output - y_true).pow(2).sum(-1).mean()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict(model, X, bs=256, dev='cuda:0'):
    yhat = torch.empty(len(X), dtype=torch.long).to(dev)

    model.eval()
    model.to(dev)
    with torch.no_grad():
        for i in range((len(X)-1)//bs + 1):
            xb = X[i*bs : i*bs+bs].to(dev)
            outputs = model(xb)
            _, preds = torch.max(outputs, dim=1)
            yhat[i*bs : i*bs+bs] = preds

    return yhat.cpu()



def toTensors(dataset : Dataset, prog=False):
    X = torch.empty(len(dataset), *dataset[0][0].shape)
    Y = torch.empty(len(dataset), dtype=torch.long)

    ds = tqdm(dataset) if prog else dataset
    for i, (x, y) in enumerate(ds):
        X[i] = x
        Y[i] = y
    return X, Y

def toTensors_dl(dataset : Dataset):
    X = torch.empty(len(dataset), *dataset[0][0].shape)
    Y = torch.empty(len(dataset), dtype=torch.long)

    bs = 2048
    dl = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=64)
    for i, (xb, yb) in enumerate(dl):
        X[i*bs: (i+1)*bs] = xb
        Y[i*bs: (i+1)*bs] = yb
    return X, Y


class TransformingTensorDataset(Dataset):
    """TensorDataset with support of torchvision transforms.
    """
    def __init__(self, X, Y, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        y = self.Y[index]

        return x, y

    def __len__(self):
        return len(self.X)


