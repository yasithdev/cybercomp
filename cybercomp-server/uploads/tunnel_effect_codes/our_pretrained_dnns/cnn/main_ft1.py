import argparse
import os
import random
import shutil
import warnings
import numpy as np
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import timm
import math
from retrieve_any_layer import ModelWrapper
from utils_cl import resnet18, LinearClassifier

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--expt_name', type=str)  # name of the experiment
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--ckpt_file', type=str, default='supervised_dnn.pth')
parser.add_argument('--save_dir', type=str, default='supervised_dnn')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--labels_dir', type=str, default='imagenet_indices')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel') #256
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay') #1e-4
parser.add_argument('--lr_step_size', default=15, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('-p', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--pretrained_task1', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained_task2', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--image_size', type=int, default=224)
best_acc1 = 0
best_acc5 = 0

def main():
    args = parser.parse_args()
    print(vars(args))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_acc5
    args.gpu = gpu
    start_time = time.time()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    ## Define arch
    model = resnet18(num_classes=args.num_classes, affine=True)

    n_parameters = sum(p.numel() for p in model.parameters())
    print('\nNumber of Parameters (in Millions):', n_parameters / 1e6)

    #print("\nThis is the created new model:")
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    ###
    #print("\nThis is the pretrained model:")
    pretrain_model1 = torch.load(args.pretrained_task1, map_location='cpu')
    state_dict1 = pretrain_model1['state_dict']
    pretrain_model2 = torch.load(args.pretrained_task2, map_location='cpu')
    state_dict2 = pretrain_model2['state_dict']
    #print("\nThis is the pre-trained model:")
    #for param_tensor in state_dict1:
    #    print(param_tensor, "\t", state_dict1[param_tensor].size())
    ###
    safe_load_dict(model, state_dict2, should_resume_all_params=False)

    reference_state_dict = pretrain_model2['state_dict']
    model_state_dict = model.state_dict()
    for k in reference_state_dict:
        assert torch.equal(model_state_dict[k[len("module."):]].cpu(), reference_state_dict[k].cpu()), k
    print("Successfully performed sanity-check on pre-trained ckpt")

    ## Split Extractor and Tunnel
    #extract_features_from="layer4.0.conv1" # 224x224 > feature dim: Nx7x7x512
    extract_features_from="layer3.1.conv2" ## 32x32 > feature dim: Nx8x8x256
    extrac = ModelWrapper(model, output_layer_names=[extract_features_from], return_single=True)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    ### define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()

    ##
    classifier = LinearClassifier(dim=256, num_class=args.num_classes) # 32x32, feature dim=8x8x256 > 256 (avg pooled & flattened)
    #classifier = LinearClassifier(dim=512, num_class=args.num_classes) # 224x224, feature dim=7x7x512 > 512 (avg pooled & flattened)
    classifier = torch.nn.DataParallel(classifier).cuda()

    # /// Optimizer /// #
    ## SGD
    #optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    ### AdamW ###
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # /// LR Schedule /// #
    ## CosineAnnealing
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    ### Cosine w/ Warmup ###
    #warmup_epochs=5
    #main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #         T_max=args.epochs - warmup_epochs, eta_min=0.0)
    #warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    #lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
    #        schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epochs])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print('\nloading the data...')
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    ####################
    labels = train_dataset.targets
    labels = np.array(labels)  ## necessary
    ## filter out only the indices for the desired class
    if args.num_classes is not None:
        train_idx = filter_by_class(labels, min_class=0, max_class=args.num_classes)
    ####################
    print('Size of training dataset:', len(train_idx))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers, 
        pin_memory=True, sampler=train_sampler, drop_last=True)

    size = int((256 / 224) * args.image_size)

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(size, interpolation=3), # to maintain same ratio w.r.t. 224 images
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ]))

    ####################
    labels = val_dataset.targets
    labels = np.array(labels)  ## necessary
    ## filter out only the indices for the desired class
    if args.num_classes is not None:
        val_idx = filter_by_class(labels, min_class=0, max_class=args.num_classes)
    ####################

    print('Size of validation dataset:', len(val_idx))
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, sampler=val_sampler)

    ####################
    ## Dataset for task-2 (50 classes)
    labels = train_dataset.targets
    labels = np.array(labels)  ## necessary
    train_idx = filter_by_class(labels, min_class=args.num_classes, max_class=2*args.num_classes)
    labels = val_dataset.targets
    labels = np.array(labels)  ## necessary
    val_idx = filter_by_class(labels, min_class=args.num_classes, max_class=2*args.num_classes)
    ####################
    print('Size of training dataset:', len(train_idx))
    train_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers, 
        pin_memory=True, sampler=train_sampler2, drop_last=True)
    print('Size of validation dataset:', len(val_idx))
    val_sampler2 = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
    val_loader2 = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, sampler=val_sampler2)

    if args.evaluate:
        acc1, acc5 = validate(val_loader, model, criterion, args, first_task=True)
        print("\nTask_1 top1 val accuracy [%]:", acc1.item(), "And Task_1 top5 val accuracy [%]:", acc5.item())

        '''
        ### Add task 2 head
        model_state_dict = model.state_dict()
        for name, param in state_dict2.items():
            if name.startswith('module.fc.'):
                #model_state_dict[name[len("module."):]].copy_(param)
                model_state_dict[name].copy_(param)
                print(name)
        print("Successfully loaded head of task 2")

        acc1, acc5 = validate(val_loader2, model, criterion, args, first_task=False)
        print("\nTask_2 top1 val accuracy [%]:", acc1.item(), "And Task_2 top5 val accuracy [%]:", acc5.item())
        '''
        return

    ### First Task
    print("\nTask_1 begins..")
    #print('starting training...')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        #loss = train(train_loader, model, criterion, optimizer, epoch, args)
        loss = train(train_loader, extrac, classifier, criterion, optimizer, epoch, args)

        lr_scheduler.step()

        ## evaluate on validation set
        #acc1, acc5 = validate(val_loader, model, criterion, args)
        acc1, acc5 = validate(val_loader, extrac, classifier, criterion, args)

    #ckpt_path = args.save_dir
    #torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
    #           f='./' + ckpt_path + '/task_1_{}'.format(args.ckpt_file))
    print('\nTop1 Accuracy After Task_1: {top1:.2f}\t'
          'Top5 Accuracy After Task_1: {top5:.2f}'.format(top1=acc1.item(), top5=acc5.item()))
    spent_time = int((time.time() - start_time) / 60)
    print("\nTotal runtime in task_1 (in minutes):", spent_time)
    exit()

    ### /////////////////////////// ###
    ### Second Task ###
    print("\nTask_2 begins..")
    ## Change / re-init head for task 2
    nn.init.kaiming_normal_(model.module.fc.weight, 'fan_in', nonlinearity='relu')
    acc1, acc5 = validate(val_loader, model, criterion, args)
    print('Top1 Accuracy After Re-init: {top1:.2f}\t'
        'Top5 Accuracy After Re-init: {top5:.2f}'.format(top1=acc1.item(), top5=acc5.item()))
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        loss = train(train_loader2, model, criterion, optimizer2, epoch, args, first_task=False)

        lr_scheduler2.step()

        # evaluate on validation set
        acc1, acc5 = validate(val_loader2, model, criterion, args, first_task=False)

    ckpt_path = args.save_dir
    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               f='./' + ckpt_path + '/task_2_{}'.format(args.ckpt_file))
    print('\nTop-1 Accuracy After Task_2: {top1:.2f}\t'
          'Top-5 Accuracy After Task_2: {top5:.2f}'.format(top1=acc1.item(), top5=acc5.item()))
    spent_time = int((time.time() - start_time) / 60)
    print("\nTotal runtime in task_2 (in minutes):", spent_time)


#def train(train_loader, model, criterion, optimizer, epoch, args, first_task=True):
def train(train_loader, extrac, classifier, criterion, optimizer, epoch, args, first_task=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion1 = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion2 = nn.CrossEntropyLoss().cuda()
    beta=1.0
    participation_rate=0.6

    # switch to train mode
    #model.train()
    extrac.cuda()
    extrac.eval()
    #classifier.cuda()
    classifier.train()
    start = time.time()
    end = time.time()

    num_iter = len(train_loader)
    loss_arr = np.zeros(num_iter, np.float32)  # Training loss

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print(f"Trained {i} batches in {time.time() - start} secs")

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if not first_task:
            target = target - args.num_classes # to remap labels
        
        if np.random.rand(1) < participation_rate:
            ### MIXUP: Do mixup between two batches of previous data
            num_instances = math.ceil(input.shape[0]/2)
            x_prev_mixed, prev_labels_a, prev_labels_b, lam = mixup_data(
                input[:num_instances], target[:num_instances],
                input[num_instances:], target[num_instances:],
                alpha=0.1)
            data = torch.empty((num_instances, input.shape[1], input.shape[2], input.shape[2]))  # mb x 80 x 14 x 14
            data = x_prev_mixed.clone()  # mb x 80 x 14 x 14
            labels_a = torch.zeros(num_instances).long()  # mb
            labels_b = torch.zeros(num_instances).long()  # mb
            labels_a = prev_labels_a
            labels_b = prev_labels_b
            #output = model(data.cuda())  # mb x 80 x 14 x 14
            output0 = extrac(data.cuda())  # mb x 80 x 14 x 14
            output = classifier(output0)
            ### Manifold MixUp
            loss = mixup_criterion(criterion1, output, labels_a.cuda(), labels_b.cuda(), lam)
            loss = loss.mean()
        else:
            ##### /// CutMix /// #######
            num_instances = math.ceil(input.shape[0]/2)
            input_a = input[:num_instances] # first 64
            input_b = input[num_instances:] # last 64
            lam = np.random.beta(beta, beta)
            target_a = target[:num_instances] # first 64
            target_b = target[num_instances:] # last 64
            bbx1, bby1, bbx2, bby2 = rand_bbox(input_a.size(), lam)
            input_a[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]
            ## adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_a.size()[-1] * input_a.size()[-2]))
            # compute output
            #output = model(input_a)
            output0 = extrac(input_a.cuda())  # mb x 80 x 14 x 14
            output = classifier(output0)
            loss = criterion2(output, target_a) * lam + criterion2(output, target_b) * (1. - lam)
            ######## /// ########

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target[:num_instances], topk=(1, 5))
        loss_arr[i] = loss.item()
        #losses.update(loss.item(), input.size(0))
        #top1.update(acc1[0], input.size(0))
        #top5.update(acc5[0], input.size(0))
        losses.update(loss.item(), num_instances)
        top1.update(acc1[0], num_instances)
        top5.update(acc5[0], num_instances)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    return np.mean(loss_arr)


def validate(val_loader, extrac, classifier, criterion, args, first_task=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    extrac.cuda()
    #classifier.cuda()
    extrac.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            if not first_task:
                target = target - args.num_classes # to remap labels

            ## compute output
            #output = model(input)
            #loss = criterion(output, target)
            output0 = extrac(input.cuda())
            output = classifier(output0)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg


def save_checkpoint(state, is_best, save_dir, ckpt_file):
    torch.save(state, os.path.join(save_dir, ckpt_file))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, ckpt_file), os.path.join(save_dir, 'best_' + ckpt_file))


def safe_load_dict(model, new_model_state, should_resume_all_params=False):
    old_model_state = model.state_dict()
    #for param_tensor in new_model_state:
    #    print(param_tensor, "\t", new_model_state[param_tensor].size())
    #print("\n")
    #for param_tensor in old_model_state:
    #    print(param_tensor, "\t", old_model_state[param_tensor].size())

    c = 0
    if should_resume_all_params:
        for old_name, old_param in old_model_state.items():
            assert old_name in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(
                old_name)
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'module':
            name = '.'.join(end)
        if name not in old_model_state:
            # print('%s not found in old model.' % name)
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            #if 'fc' not in name:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')


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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def filter_by_class(labels, min_class, max_class):
    return list(np.where(np.logical_and(labels >= min_class, labels < max_class))[0])

def mixup_data(x1, y1, x2, y2, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * x1 + (1 - lam) * x2
    y_a, y_b = y1, y2
    return mixed_x, y_a, y_b, lam

# Mix Up Criterion #
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a.squeeze()) + (1 - lam) * criterion(pred, y_b.squeeze())

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


if __name__ == '__main__':
    main()