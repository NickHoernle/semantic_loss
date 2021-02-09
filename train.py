import argparse
import os
import shutil
import time
import git

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from wideresnet import WideResNet

from symbolic import *

# used for logging to TensorBoard

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--dataset_path', default='../data', type=str,
                    help='path to where the data are stored')
parser.add_argument("--checkpoint_dir", default="runs")
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--no-sloss', dest='sloss', action='store_false',
                    help='whether to use semantic logic loss (default: True)')
parser.add_argument('--no-superclass', dest='superclass', action='store_false',
                    help='whether to test on baseline for superclass accuracy')
parser.set_defaults(augment=True)
parser.set_defaults(sloss=True)

best_prec1 = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
repo = git.Repo(search_parent_directories=True)
git_commit = repo.head.object.hexsha

def main():
    global args, best_prec1, class_ixs, sloss, params

    args = parser.parse_args()
    sloss = args.sloss
    superclass = False
    print(sloss, superclass)
    params = f"{args.name}_{sloss}_{args.lr}"

    if args.tensorboard: configure(os.path.join(args.checkpoint_dir, git_commit, params))
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](args.dataset_path, train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](args.dataset_path, train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    num_classes = (args.dataset == 'cifar10' and 10 or 100)
    class_ixs = get_class_ixs(args.dataset)
    if sloss:
        print("Testing model")
        terms = get_logic_terms(args.dataset)
        model = ConstrainedModel(args.layers, num_classes, terms, args.widen_factor, dropRate=args.droprate)
    elif superclass:
        exp_params = get_experiment_params(args.dataset)
        model = WideResNet(args.layers, exp_params["num_classes"], args.widen_factor, dropRate=args.droprate)
    else:
        print("Baseline model")
        model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=.2)


    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1a = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)

        if sloss:
            class_preds, logic_preds = output
            ll = []
            for j, p in enumerate(class_preds.split(1, dim=1)):
                ll += [F.cross_entropy(p.squeeze(1), target, reduction="none")]

            pred_loss = torch.stack(ll, dim=1)
            recon_losses, labels = pred_loss.min(dim=1)

            loss = (logic_preds.exp() * (pred_loss + logic_preds)).sum(dim=1).mean()
            loss += recon_losses.mean()

            class_pred = class_preds[np.arange(len(target)), logic_preds.argmax(dim=1)]

        else:
            class_pred = output
            loss = criterion(class_pred, target)
        # measure accuracy and record loss
        prec1 = accuracy(class_pred.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        losses.update(loss.data.item(), input.size(0))

        # get the super class accuracy
        # new_tgts = torch.zeros_like(target)
        # for i, ixs in enumerate(class_ixs[1:]):
        #     new_tgts += (i+1)*(torch.stack([target == i for i in ixs], dim=1).any(dim=1))
        #
        # forward_mapping = [int(c) for ixs in class_ixs for c in ixs]
        # split = class_pred.log_softmax(dim=1)[:, forward_mapping].split([len(i) for i in class_ixs], dim=1)
        # new_pred = torch.stack([s.logsumexp(dim=1) for s in split], dim=1)
        #
        # prec_1a = accuracy(new_pred.data, new_tgts, topk=(1,))[0]
        # top1a.update(prec_1a.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'PrecSG@1 {top1a.val:.3f} ({top1a.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1, top1a=top1a))

    if epoch % 5 == 4:
        if sloss:
            model.threshold1p()

    # if sloss:
    #     if epoch % 5 == 4:
    #         model.threshold1p()
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1a = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input, test=True)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # get the super class accuracy
        new_tgts = torch.zeros_like(target)
        for j, ixs in enumerate(class_ixs[1:]):
            new_tgts += (j + 1) * (torch.stack([target == k for k in ixs], dim=1).any(dim=1))

        forward_mapping = [int(c) for ixs in class_ixs for c in ixs]
        split = output.log_softmax(dim=1)[:, forward_mapping].split([len(k) for k in class_ixs], dim=1)
        new_pred = torch.stack([s.logsumexp(dim=1) for s in split], dim=1)

        prec_1a = accuracy(new_pred.data, new_tgts, topk=(1,))[0]
        top1a.update(prec_1a.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'PrecSG@1 {top1a.val:.3f} ({top1a.avg:.3f})'.format(
                epoch, i, len(val_loader), batch_time=batch_time,
                loss=losses, top1=top1, top1a=top1a))

    print(f'{epoch} * Prec@1 {round(top1.avg, 3)}, '
          f'PrecSG@1 {round(top1a.avg, 3)} ({int(top1a.sum)}/{top1a.count}), '
          f'Loss {round(losses.avg, 3)}')
    # log to TensorBoard
    if args.tensorboard:
        from tensorboard_logger import configure, log_value
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = os.path.join(args.checkpoint_dir, git_commit, params)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fname = os.path.join(directory, filename)
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, os.path.join(directory, "best_" + filename))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
