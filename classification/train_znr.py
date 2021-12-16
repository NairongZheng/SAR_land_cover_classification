"""
    date:2021.11.29
"""
import os
import argparse
from abc import ABC
import cv2
import numpy as np
import time
import shutil
from PIL import Image
import random
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data
import torchvision.models as models

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"

best_acc1 = 0

def NMT(img, img_mean, rate=2.5, EPS=1e-7):
    img[img > (rate * img_mean + EPS)] = rate * img_mean
    img = (img / (np.max(img) + EPS)) * 255.
    img = img.astype('uint8')

    return img

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.1 ** (epoch // 10))
    # lr = args.lr * (0.5 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, acc1, top1_acc, file):
    file = os.path.join(file, 'output_112_80p_resnet50_32x4d')
    if not os.path.exists(file):
        os.makedirs(file)
    file_name = str(state['arch']) + '_epoch' + str(state['epoch']) + '_TrainAcc' + str(top1_acc.cpu().numpy())[:5] + '_ValAcc' + str(acc1.cpu().numpy())[:5] + '_ckpt.pth.tar'
    filename = os.path.join(file, file_name)
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(file, 'model_best.pth.tar')
        shutil.copyfile(filename, best_file)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def make_datasets(dir, class_to_idx):
    instances = []
    dir = os.path.expanduser(dir)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(dir, target_class)
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)
    return instances

class ClassificationDatasetLoader(data.Dataset, ABC):
    def __init__(self, root, transform=True, is_flip=True):
        super(ClassificationDatasetLoader, self).__init__()

        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of : {}\n".format(self.root)
            raise RuntimeError(msg)
        self.samples = samples
        self.transform = transform
        self.is_flip = is_flip
    
    def _find_classes(slef, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        return make_datasets(dir, class_to_idx)

    def input_transform(self, image):
        # image = image.astype(np.float32)[:, :, ::-1]  # 通道逆序, why?
        # image = image.astype(np.float32)[:, :, np.newaxis]
        # image = np.tile(image, (1, 1, 3))
        image = image.astype(np.float32)
        image = image / 127.5 - 1  # [-1, 1]
        # image = image / 255.
        # image -= self.mean
        # image /= self.std
        return image

    def image_resize(self, image, new_w=224, new_h=224):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return image

    def gen_samples(self, image, is_flip=True):
        image = self.image_resize(image)
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]      # 随机选择是否需要水平翻转
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）

        return image

    def __getitem__(self, index):
        path, target = self.samples[index]

        # 因为数据没有经过处理，还是single类型，所以这边要加个nmt然后再转uint8!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 如果数据是提前处理好的，就直接用cv2读就可以了
        # sample = cv2.imread(path, cv2.IMREAD_COLOR)  # color - 3, grayscale - 1
        # sample = sample.astype(np.float32)[:, :, ::-1]  # BRG -> RGB

        img = Image.open(path)
        img = np.array(img, dtype='float32')
        img_mean = 0.308
        img = NMT(img, img_mean)
        img = img.astype(np.float32)[:, :, np.newaxis]
        sample = np.tile(img, (1, 1, 3))

        if not self.transform:
            sample = self.input_transform(sample)
            sample = sample.transpose((2, 0, 1))  # [c, h, w]
        else:
            sample = self.gen_samples(sample, self.is_flip)

        return sample.copy(), np.array(target)

    def __len__(self):
        return len(self.samples)

def parse_args(model_names):
    """
        initialize parameter
    """
    parser = argparse.ArgumentParser(description='train classification network')

    # data and model
    parser.add_argument('--data', help='path to dataset', 
                    default='/emwuser/znr/code/micro_wave/classification/data',
                    type=str, metavar='DIR')
    parser.add_argument('--arch', help='model architecture: ' + ' | '.join(model_names), default='densenet121',
                    metavar='ARCH', choices=model_names)

    # DDP options
    parser.add_argument('--nodes', help='the number of nodes for distributed training', default=1, type=int)        # 要使用的节点数
    # parser.add_argument('--gpus', help='the number of gpus per node', default=4, type=int)                          # 每个节点上的GPU数量(直接用torch看就可以)
    parser.add_argument('--node_rank', help='ranking within the nodes', default=0, type=int)                        # 当前节点在所有节点中的排名
    parser.add_argument('--local_rank', help='local_rank', default=0, type=int)
    parser.add_argument('--dist_backend', help='distributed backend', default='nccl', type=str)
    parser.add_argument('--dist_url', help='url used to set up distributed training', default='tcp://127.0.0.1:8046', type=str)
    parser.add_argument('--multiprocessing_distributed', action='store_true', default=True, 
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--gpu', help='the gpu used', default=0, type=int)

    # training
    parser.add_argument('--pretrained', help='use pretrained model', default=True, type=bool)
    parser.add_argument('--classes', help='the number of classes', default=10, type=int)
    parser.add_argument('--epochs', help='the number of total epochs to run', default=20, type=int)
    parser.add_argument('--start_epoch', help='manual epoch number (useful on restarts)', default=0, type=int)
    parser.add_argument('--batch_size', help='mini-batch size', default=32)
    parser.add_argument('--lr', help='learning rate', default=0.1, type=float)
    parser.add_argument('--momentum', help='the momentum of optimizer', default=0.9, type=float)
    parser.add_argument('--weight_decay', help='the weight_decay of optimizer', default=1e-4, type=float)

    # settings
    parser.add_argument('--resume', help='path to latest checkpoint', default='', type=str)
    parser.add_argument('--print_freq', help='print frequency', default=60, type=int)
    parser.add_argument('--evaluate', help='evaluate model on validation dataset', default=False, type=bool)

    args = parser.parse_args()
    return args


def main():
    """
        主函数
    """
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
    args = parse_args(model_names)

    ngpus_per_node = torch.cuda.device_count()

    # judge if using distributed training or not
    args.distributed = args.nodes > 1 or args.multiprocessing_distributed

    if args.distributed:
        args.world_size = ngpus_per_node * args.nodes       # 总进程数=每个节点的GPU数量*节点数
        print('=> use distributed training, world_size:{}'.format(args.world_size))
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print('=> do not use distributed training')
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    """
        main_worker
    """
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.node_rank * ngpus_per_node + gpu      # 进程编号=机器编号*每台机器可用GPU数量+当前GPU编号(这里的gpu其实就是local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))  # model name
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    
    # 修改最后一层输出
    if args.arch == 'resnet34':         # lr=0.1
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, args.classes)

    # vgg16
    elif args.arch == 'vgg16':          # lr=0.001
        # model.features[0] = nn.Conv2d(ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=args.classes, bias=True)

    # densenet121
    elif args.arch == 'densenet121':    # lr=0.1
        # model.features.conv0 = nn.Conv2d(ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = nn.Linear(in_features=1024, out_features=args.classes, bias=True)

    elif args.arch == 'resnext50_32x4d':    # lr=0.1
        fc_features = model.fc.in_features
        model.fc = nn.Linear(in_features=fc_features, out_features=args.classes, bias=True)

    print(model)

    # define optimozer and criterion
    class_weights = None
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.resume:  # checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    
    # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    cudnn.benchmark = True

    # loading data
    train_dir = os.path.join(args.data, 'Train_classification_112_80p')
    val_dir = os.path.join(args.data, 'Test_classification_112_80p')

    train_dataset = ClassificationDatasetLoader(train_dir)
    val_dataset = ClassificationDatasetLoader(val_dir)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.batch_size, shuffle=(train_sampler is None),
                                            num_workers=0, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)
    
    
    if args.evaluate:  # test
        validate(val_loader, model, criterion, args)
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        top1_acc, rank = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #                                             and args.rank % ngpus_per_node == 0):
        if rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, acc1, top1_acc, args.data)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    rank = get_rank()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.gpu == 0:
            progress.display(i)

    return top1.avg, rank

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        # [batch_time, losses, top1, top5],
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    main()
