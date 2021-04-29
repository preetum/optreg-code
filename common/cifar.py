import torch
import torchvision
import torchvision.datasets as dst
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm.auto import tqdm, trange
import numpy as np

from common import toTensors

def get_data_aug_transform():
    """
        Returns a torchvision transform that maps (normalized Tensor) --> (normalized Tensor)
        via a random data augmentation.
    """
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    unnormalize = transforms.Compose([
        transforms.Normalize((0, 0, 0), (2, 2, 2)),
        transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1))
    ])

    return transforms.Compose(
        [unnormalize,
         transforms.ToPILImage(),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize
         ])

def load_cifar(normalize=True):
    train_ds = dst.CIFAR10(root='~/tmp/data', train=True,
                           download=True, transform=None)
    test_ds = dst.CIFAR10(root='~/tmp/data', train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        X = torch.Tensor(np.transpose(dataset.data, (0, 3, 1, 2))).float() / 255.0
        if normalize:
            X = 2.0*X - 1.0 # [0, 1] --> [-1, 1]
        Y = torch.Tensor(dataset.targets).long()
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te

def load_cifar_old(data_aug = True, noise_aug = False, noise_L2 = 1.0):
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train = transforms.Compose(
        [#torchvision.transforms.ToPILImage(),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if noise_aug:
        # Add Gaussian Noise, with Variance = (noise_L2)^2
        transform_train = transforms.Compose(
            [transform_train,
                Lambda(lambda batch: batch + noise_L2*torch.randn_like(batch)/np.sqrt(3*32*32))
            ])


    all_test_ds = torchvision.datasets.CIFAR10(root='~/data', train=False,
                                           download=True, transform=transform_test)


    X_te, Y_te = toTensors(all_test_ds)

    if data_aug:
        tr_ds = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                            download=True, transform=transform_train)

    else:
        tr_ds = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                            download=True, transform=transform_test)

    return tr_ds, X_te, Y_te

def cifar_labels():
    return ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
