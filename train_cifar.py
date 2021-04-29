import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm.auto import tqdm, trange
from argparse import ArgumentParser
import numpy as np
from PIL import Image

from common import gsave, dload, gload, gopen, predict, toTensors, TransformingTensorDataset, count_parameters, toTensors_dl
from common.logging import Logger


# Import models
from common.models.myrtle_cnn import mCNN_bn_k
from common.models.ResNet18k import resnet18k_cifar, resnet50k_cifar

import wandb

def update_summary(metrics):
    for k, v in metrics.items():
        wandb.run.summary[f"Final {k}"] = v


def evaluate_ds(model, ds : Dataset, bsize=128, loss_func = nn.CrossEntropyLoss().cuda()):
    ''' Returns loss, acc'''
    test_dl = DataLoader(ds, batch_size=bsize, shuffle=False, num_workers=8)

    model.eval()
    model.cuda()
    nCorrect = 0.0
    nTotal = 0
    net_loss = 0.0
    with torch.no_grad():
        for (xb, yb) in tqdm(test_dl):
            xb, yb = xb.cuda(), yb.cuda()
            outputs = model(xb)
            loss = len(xb)*loss_func(outputs, yb)
            _, preds = torch.max(outputs, dim=1)
            nCorrect += (preds == yb).float().sum()
            net_loss += loss
            nTotal += preds.size(0)

    acc = nCorrect.cpu().item() / float(nTotal)
    loss = net_loss.cpu().item() / float(nTotal)
    return loss, acc

def set_lr(opt, lr):
    ''' Sets the learning rate of optimizer OPT '''
    for g in opt.param_groups:
        g['lr'] = lr

def fit(args, logger, model, train_ds, test_sets, opt, start_epoch=0, epoches=100, bsize=128,
            loss_func = nn.CrossEntropyLoss().cuda(),
            loss_thresh = None,
            lr_func = None):
    if not args.fast:
        test_sets['Train'] = train_ds

    train_dl = DataLoader(train_ds, batch_size=bsize, shuffle=False, num_workers=8)
    model.cuda()
    
    for epoch in range(start_epoch, start_epoch+epoches):
        lr = lr_func(epoch)
        set_lr(opt, lr)

        nCorrect = nTotal = netLoss = 0.0
        model.train()
        for (xB, yB) in tqdm(train_dl):
            opt.zero_grad()
            xB, yB = xB.cuda(), yB.cuda()
            outputs = model(xB)
            loss = loss_func(outputs, yB)
            loss.backward()
            opt.step()

            with torch.no_grad():
                netLoss += len(xB)*loss.data
                _, preds = torch.max(outputs, dim=1)
                nCorrect += (preds == yB).float().sum()
                nTotal += preds.size(0)


        metrics = {'Epoch' : epoch}
        for te_name, te_ds in test_sets.items():
            loss, acc = evaluate_ds(model, te_ds, bsize=args.eval_bsize, loss_func=loss_func)
            metrics[te_name + " Loss"] = loss
            metrics[te_name + " Error"] = 1.0-acc
        if lr_func is not None:
            metrics['lr'] = lr
        
        if args.fast:
            train_acc = (nCorrect/nTotal).item()
            train_loss = (netLoss / nTotal).item()
            metrics.update({
                "Train Error": 1.0-train_acc,
                "Train Loss": train_loss})

        gsave(model.state_dict(), "gs://preetum/models/%s-n%d/wandb_%s/ep%d" % (args.proj_name, wandb.config.nSamps, wandb.run.id, epoch))
        print('Epoch %d.\t Train Loss: %.3f \t Train Acc: %.3f \t Test Acc: %.3f' %
            (epoch, metrics['Train Loss'], 1.0-metrics['Train Error'], 1.0-metrics['Test Error']))

        wandb.log(metrics, step=epoch)
        update_summary(metrics)

        logger.log(metrics)
        logger.log_summary(metrics)
        logger.sync()

        if loss_thresh is not None and metrics['Train Loss'] <= loss_thresh:
            # stop optimization when train loss is small enough
            return metrics

    return metrics

def fit_steps(args, logger, model, train_ds, test_sets, opt, start_step, nsteps=1000000, bsize=128,
            fast = True,
            loss_func = nn.CrossEntropyLoss().cuda(),
            loss_thresh = None,
            lr_func=None,
            step_log_interval=512):
    if not args.fast:
        test_sets['Train'] = train_ds

    train_dl = DataLoader(train_ds, batch_size=bsize, shuffle=True, num_workers=8)
    
    model.cuda()

    step = start_step
    lr = lr_func(step // 512)
    set_lr(opt, lr)

    nCorrect = nTotal = netLoss = 0.0
    while True:
        for (xB, yB) in tqdm(train_dl):

            model.train()
            opt.zero_grad()
            xB, yB = xB.cuda(), yB.cuda()
            outputs = model(xB)
            loss = loss_func(outputs, yB)
            loss.backward()
            opt.step()
            step += 1

            with torch.no_grad():
                netLoss += len(xB)*loss.data
                _, preds = torch.max(outputs, dim=1)
                nCorrect += (preds == yB).float().sum()
                nTotal += preds.size(0)


            if step >= start_step + nsteps:
                return

            if step % 512 == 0:
                lr = lr_func(step // 512)
                set_lr(opt, lr)
                gsave(model.state_dict(), "gs://preetum/models/%s-n%d/wandb_%s/step%d" % (args.proj_name, wandb.config.nSamps, wandb.run.id, step))

            if step % step_log_interval == 0:

                metrics = {'Step' : step}
                for te_name, te_ds in test_sets.items():
                    loss, acc = evaluate_ds(model, te_ds, bsize=args.eval_bsize, loss_func=loss_func)
                    metrics[te_name + " Loss"] = loss
                    metrics[te_name + " Error"] = 1.0-acc
                metrics['lr'] = lr

                if args.fast:
                    train_acc = (nCorrect/nTotal).item()
                    train_loss = (netLoss / nTotal).item()
                    metrics.update({
                        "Train Error": 1.0-train_acc,
                        "Train Loss": train_loss})
                    nCorrect = nTotal = netLoss = 0.0

                print('Step %d.\t Train Loss: %.3f \t Train Acc: %.3f \t Test Acc: %.3f' %
                    (step, metrics['Train Loss'], 1.0-metrics['Train Error'], 1.0-metrics['Test Error']))


                wandb.log(metrics)
                update_summary(metrics)

                logger.log(metrics)
                logger.log_summary(metrics)
                logger.sync()

                if loss_thresh is not None and metrics['Train Loss'] <= loss_thresh:
                    # stop optimization when train loss is small enough
                    return

    return




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

def get_fliponly_aug_transform():
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
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize
         ])


def train(args):
    ## GCS Logging
    logger = Logger(proj_name = args.proj_name, wandb=wandb)

    ## Load all train and test sets
    print("Loading Dataset...")
    if args.dataset == 'cifar10':
        X_tr_all, Y_tr_all, X_te, Y_te = dload("gs://preetum/datasets/cifar10")
        num_classes = 10
    if args.dataset == 'cifar100':
        X_tr_all, Y_tr_all, X_te, Y_te = dload("gs://preetum/datasets/cifar100")
        num_classes = 100

    def add_noise(Y, p: float):
        ''' Adds noise to Y, s.t. the label is wrong w.p. p '''
        noise_dist = torch.distributions.categorical.Categorical(
            probs=torch.tensor([1.0 - p] + [p / (num_classes-1)] * (num_classes-1)))
        return (Y + noise_dist.sample(Y.shape)) % num_classes

    # take nSamps random training examples (+ label noise)
    print("Subsampling/Noising Dataset...")
    I = np.random.permutation(len(X_tr_all))
    X_tr_all = X_tr_all[I]
    Y_tr_all = Y_tr_all[I]
    X_tr, Y_tr = X_tr_all[:args.nSamps], Y_tr_all[:args.nSamps]
    Y_tr = add_noise(Y_tr, args.noise) # label noise
    if not args.clean_test:
        Y_te = add_noise(Y_te, args.noise) # measure noisy test error

    
    ## Init model
    if args.model == 'mCNNk_bn':
        model = mCNN_bn_k(args.k, num_classes=num_classes) # with BatchNorm
    elif args.model == 'resnet18k':
        model = resnet18k_cifar(num_classes=num_classes, k=args.k)
    elif args.model == 'resnet50k':
        model = resnet50k_cifar(num_classes=num_classes, k=args.k)
    else:
        print("Unrecognized Model: ", args.model)
        exit()

    nparams = count_parameters(model)
    print('Model parameters: ', nparams)
    wandb.run.summary[f"Model Params"] = nparams

        

    ## Data Sets

    test_sets = {'Test' : TensorDataset(X_te, Y_te)}

    # DataAug Test/Train
    if args.data_aug:
        transform = get_data_aug_transform()
        train_ds = TransformingTensorDataset(X_tr, Y_tr, transform)
        if not args.fast:
            test_sets['AugTest'] = TransformingTensorDataset(X_te, Y_te, transform)
    else:
        train_ds = TensorDataset(X_tr, Y_tr)

    if args.fliponly_aug: # special-purpose flag
        transform = get_fliponly_aug_transform()
        train_ds = TransformingTensorDataset(X_tr, Y_tr, transform)

    ## Learning Rate Schedule, and opt
    if args.opt == 'sgd':
        lr_sched = [(0.1, 1000)]
        opt_fam = optim.SGD

    if args.opt == 'adam':
        lr_sched = [(0.0001, 10000)] #[(0.01, 50), (0.001, 200), (0.0001, 200)]         
        opt_fam = optim.Adam

    if args.lr is not None:
        lr_sched = eval(args.lr) # pull the learning rate schedule from arguments if specified
        print("Learning Rate Schedule: ", lr_sched)

    wd = args.decay
    # if args.decay:
    #     wd = 5*1e-4

    ## Loss Functions
    def mse_loss(output, y):
        y_true = F.one_hot(y, 10).float()
        return (output - y_true).pow(2).sum(-1).mean()
    if args.loss == 'ce':
        loss_func = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'mse':
        loss_func = mse_loss
    elif args.loss == 'smse':
        # softmax + MSE
        loss_func = lambda output, y: mse_loss(F.softmax(output), y)


    if not args.steps:
        # LR schedule in terms of epoches (standard)
        ep = 0
        for (lr, nEps) in lr_sched:
            opt = opt_fam(model.parameters(), lr=lr, weight_decay=wd)

            if args.lrfunc == 'invsqrt':
                def invsqrt_lr(epoch: int):
                    return lr/np.sqrt(epoch+1)
                lr_func = invsqrt_lr
            else:
                def lr_func(_): return lr


            fit(args, logger, model, train_ds, test_sets, opt, bsize = args.bsize, start_epoch = ep,
                epoches=nEps, loss_thresh=args.loss_thresh, loss_func=loss_func, lr_func=lr_func)
            ep += nEps
    else:
        # LR schedule in terms of gradient steps
        steps = 0
        for (lr, nsteps) in lr_sched:
            opt = opt_fam(model.parameters(), lr=lr, weight_decay=wd)

            if args.lrfunc == 'invsqrt':
                def invsqrt_lr(epoch: int):
                    return lr/np.sqrt(epoch+1)
                lr_func = invsqrt_lr
            else:
                def lr_func(_): return lr

            fit_steps(args, logger, model, train_ds, test_sets, opt, bsize = args.bsize, start_step=steps,
                nsteps=nsteps, loss_thresh=args.loss_thresh, loss_func=loss_func, lr_func=lr_func,
                step_log_interval=args.step_log_interval)
            steps += nsteps


    logger.log_summary({'finished': True})
    logger.sync()

    gsave(model.state_dict(), "gs://preetum/models/%s/wandb_%s" % (args.proj_name, wandb.run.id))
    return

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--nSamps', type=int, default=50000, help="Number of training examples to use.")
    ap.add_argument('--lr', type=str)
    ap.add_argument('--lrfunc', default='const', choices=['const', 'invsqrt'])
    ap.add_argument('--model', type=str, default='mCNNk_bn', choices=['mCNNk_bn', 'resnet18k', 'resnet50k'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--noise', type=float, default=0.0)
    ap.add_argument('--bsize', type=int, default=128, help="Train Batch size")
    ap.add_argument('--eval_bsize', type=int, default=256, help="Evaluation Batch size")
    ap.add_argument('--proj_name', type=str, default="cifar-test-2")
    ap.add_argument('--loss_thresh', type=float, default=0.00000)
    ap.add_argument('--loss', type=str, default='ce', choices = ['ce', 'mse', 'smse'])
    ap.add_argument('--opt', type=str, default='adam', choices=['sgd', 'adam'])
    ap.add_argument('--data_aug', action='store_true')
    ap.add_argument('--fliponly_aug', action='store_true')
    ap.add_argument('--clean_test', action='store_true', help="Measure CLEAN test error.")
    ap.add_argument('--fast', action='store_true', help='Just average the losses for Train Loss, and dont compute AugTest Error')
    ap.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    ap.add_argument('--steps', action='store_true', help='LR schedule in terms of gradient steps, not epoches.')
    ap.add_argument('--step_log_interval', type=int, default=512)
    ap.add_argument('--decay', type=float, default=0.0)

    args = ap.parse_args()

    wandb.init(project=args.proj_name)
    wandb.config.update(args, allow_val_change=True)

    train(args)
