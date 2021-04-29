import torch.nn as nn

from common import gsave, gload

def load_dict(model, d):
    '''Loads state-dict d into model (wrapping and unwrapping model in DataParallel if neccesary)'''
    isDataParallel = list(d.keys())[0].startswith('module')
    if isDataParallel:
        model = nn.DataParallel(model)
        model.load_state_dict(d)
        return model.module
    else:
        model.load_state_dict(d)
        return model

