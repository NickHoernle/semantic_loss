import os
import git

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from datasets import SyntheticDataset, ConstraintedSampler
from symbolic import *

device = "cuda" if torch.cuda.is_available() else "cpu"
repo = git.Repo(search_parent_directories=True)
git_commit = repo.head.object.hexsha

tensorboard = False

def main():

    # params = f"{args.layers}_{args.widen_factor}_{sloss}_{args.lr}_{args.ll}_{args.ul}"
    params = "synthetic"
    print(params)

    if tensorboard: configure(os.path.join(args.checkpoint_dir, git_commit, params))

    dset_params = {
        "sampler": ConstraintedSampler,

    }
    dataset = SyntheticDataset()
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # create model
    num_classes = (args.dataset == 'cifar10' and 10 or 100)
    class_ixs = get_class_ixs(args.dataset, classes)
    if sloss:
        print("Testing model")
        terms = get_logic_terms(args.dataset, classes, args.ll, args.ul, device=device)
        map = get_map_matrix(args.dataset, classes).to(device)
        model = ConstrainedModel(args.layers, num_classes, terms, args.widen_factor,
                                 dropRate=args.droprate)
    elif superclass:
        exp_params = get_experiment_params(args.dataset, classes)
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
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=.5)

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

    # load the best model and evaluate on the test set
    print("======== TESTING ON UNSEEN DATA =========")
    print("======== USE FINAL MODEL =========")
    prec1 = validate(test_loader, model, criterion, 0)
    print('Final Model accuracy ====> ', prec1)
    print("======== USE BEST MODEL =========")
    directory = os.path.join(args.checkpoint_dir, git_commit, params, "best_checkpoint.pth.tar")
    checkpoint = torch.load(directory)
    model.load_state_dict(checkpoint['state_dict'])
    prec1 = validate(test_loader, model, criterion, 0)
    print('Test accuracy ====> ', prec1)