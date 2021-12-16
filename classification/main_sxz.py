import argparse
import os
import random
import shutil
import time
import warnings
import math
from abc import ABC

import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from PIL import Image
# import os
import os.path
import numpy as np
from torch.utils import data


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


def NMT(img, img_mean, rate=2.5, EPS=1e-7):
    img[img > (rate * img_mean + EPS)] = rate * img_mean
    img = (img / (np.max(img) + EPS)) * 255.
    img = img.astype('uint8')

    return img


class CIFAR10(data.Dataset, ABC):
    def __init__(self, root, transform, base_size=(224, 224), crop_size=(112, 112), multi_scale=True,
                 is_flip=True, center_crop_test=False, scale_factor=16):

        super(CIFAR10, self).__init__()

        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of : {}\n".format(self.root)
            raise RuntimeError(msg)
        self.samples = samples

        self.transform = transform
        # self.target_transform = target_transform
        self.multi_scale = multi_scale
        self.is_flip = is_flip
        self.center_crop_test = center_crop_test
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def input_transform(self, image):
        # image = image.astype(np.float32)[:, :, ::-1]  # 通道逆序, why?
        image = image.astype(np.float32)[:, :, np.newaxis]
        image = np.tile(image, (1, 1, 3))
        image = image / 127.5 - 1  # [-1, 1]
        # image = image / 255.
        # image -= self.mean
        # image /= self.std
        return image

    def pad_image(self, image, h, w, size, pad_value):
        pad_image = image.copy()  # shadow copy
        pad_h = max(size[0] - h, 0)  # 判断是否需要填充  [h, w, c]
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:  # 右下方填充
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)  # 边框

        return pad_image

    def rand_crop(self, image):  # , label
        h, w = image.shape[:-1]  # [h, w, c]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))  # pad, 填充零值
        new_h, new_w = image.shape[:-1]  # [h, w], mask
        x = random.randint(0, new_w - self.crop_size[1])  # w, col
        y = random.randint(0, new_h - self.crop_size[0])  # h, row
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image

    def center_crop(self, image):  #
        h, w = image.shape[:-1]
        x = int(round((w - self.crop_size[1]) / 2.))  # w
        y = int(round((h - self.crop_size[0]) / 2.))  # h
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image

    def image_resize(self, image, long_size):
        h, w = image.shape[:-1]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return image

    def multi_scale_aug(self, image, rand_scale=1., rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        image = self.image_resize(image, long_size)
        if rand_crop:
            image = self.rand_crop(image)

        return image

    def gen_samples(self, image, multi_scale=True, is_flip=True, center_crop_test=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image = self.multi_scale_aug(image, rand_scale=rand_scale)
        if center_crop_test:
            image = self.image_resize(image, self.base_size)
            image = self.center_crop(image)

        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）

        return image

    @staticmethod
    def make_dataset(dir, class_to_idx):
        return make_datasets(dir, class_to_idx)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample = cv2.imread(path, cv2.IMREAD_COLOR)
        if not self.transform:
            sample = self.input_transform(sample)
            sample = sample.transpose((2, 0, 1))  # [c, h, w]
        else:
            sample = self.gen_samples(sample, self.multi_scale, self.is_flip, self.center_crop_test)

        return sample, target

    def __len__(self):
        return len(self.samples)


class FUSARMap(data.Dataset, ABC):
    def __init__(self, root, transform=True, is_flip=True):

        super(FUSARMap, self).__init__()

        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of : {}\n".format(self.root)
            raise RuntimeError(msg)
        self.samples = samples
        self.transform = transform
        self.is_flip = is_flip

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
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）

        return image

    @staticmethod
    def make_dataset(dir, class_to_idx):
        return make_datasets(dir, class_to_idx)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample = cv2.imread(path, cv2.IMREAD_COLOR)  # color - 3, grayscale - 1
        sample = sample.astype(np.float32)[:, :, ::-1]  # BRG -> RGB
        if not self.transform:
            sample = self.input_transform(sample)
            sample = sample.transpose((2, 0, 1))  # [c, h, w]
        else:
            sample = self.gen_samples(sample, self.is_flip)

        return sample.copy(), np.array(target)

    def __len__(self):
        return len(self.samples)


class FUSARMapV2(data.Dataset, ABC):
    def __init__(self, root, transform=True, is_flip=True):

        super(FUSARMapV2, self).__init__()

        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        print(classes)
        samples = self.make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of : {}\n".format(self.root)
            raise RuntimeError(msg)
        self.samples = samples
        self.transform = transform
        self.is_flip = is_flip

    def input_transform(self, image):
        # image = image.astype(np.float32)[:, :, ::-1]  # 通道逆序, why?
        # image = image.astype(np.float32)[:, :, np.newaxis]  # [h, w, c]
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
            flip = np.random.choice(2) * 2 - 1  # [0, 1] * 2 - 1 = [-1, 1]
            image = image[:, :, ::flip]  # channel [-1, 1] 正向（不变），反向（倒序）

        return image

    @staticmethod
    def make_dataset(dir, class_to_idx):
        return make_datasets(dir, class_to_idx)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        # sample = self.loader(path)
        # sample = cv2.imread(path, cv2.IMREAD_COLOR)  # color - 4
        img = Image.open(path)
        img = np.array(img, dtype='float32')
        img_mean = 0.308
        img = NMT(img, img_mean)
        img = img.astype(np.float32)[:, :, np.newaxis]
        sample = np.tile(img, (1, 1, 3))
        # sample = sample.astype(np.float32)
        # sample = sample.astype(np.float32)[:, :, ::-1]  # bgr -> rgb

        if not self.transform:
            sample = self.input_transform(sample)
            sample = sample.transpose((2, 0, 1))  # [c, h, w]
        else:
            sample = self.gen_samples(sample, self.is_flip)

        return sample.copy(), np.array(target)

    def __len__(self):
        return len(self.samples)


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


os.environ['CUDA_VISIBLE_DEVICES'] = "6, 7"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

"""
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:8001' --dist-backend 'nccl'

python main.py -a resnet50 

"""

# training/validation datasets
# parser.add_argument('-d', '--data',
#                     default='/data/shixianzheng/2021_research/SAR_classification/FUSAR-Map-V2/singlePolSAR/',
#                     metavar='DIR',
#                     help='path to dataset')

parser.add_argument('-d', '--data',
                    default='/emwuser/znr/code/micro_wave/classification/data',
                    metavar='DIR',
                    help='path to dataset')

# network
# resnet34, vgg16, densenet121, resnext50-32x4d
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet34)')

# loading workers for sampler
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# epoch training, batch-size
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# learning rate (SGD)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,   # vgg 0.01
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# setting parameters
parser.add_argument('-p', '--print-freq', default=60, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# pre-trained
parser.add_argument('--pretrained', default=False, type=bool,
                    help='use pre-trained model')

# distributed training
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8003', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

# seed for re-implement
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
#                     help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
# parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

best_acc1 = 0
classes = 10
# ch = 3


def main():
    args = parser.parse_args()

    # 保证训练的可重复
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
        # torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU.
        # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
        cudnn.deterministic = True  # 每次返回的卷积算法将是确定的，即默认算法
        # torch.cuda.manual_seed_all(seed) Sets the seed for generating random numbers on all GPUs.
        # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # 指定GPU
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # distributed training
    # distributed 相较于 DataParallel，速度更快，效率更高
    # world size: 代表全局进程的个数
    # rank: 表示进程序号，用于进程间通讯，表征进程优先级
    # local rank: 进程内，GPU编号
    if args.dist_url == "env://" and args.world_size == -1:  # 使用 init_process_group 初始化进程组
        args.world_size = int(os.environ["WORLD_SIZE"])

    # 分布式训练判断
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 返回可用的GPU数量， node表示主机
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:  # DDP？
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size  # 多机多卡，进程总数
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1  # 全局变量
    args.gpu = gpu

    # 可利用的GPU
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # 分布式训练进程组初始化
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))  # model name
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # print(model)
    # 修改类别数,3, 10
    # resnet
    # model.conv1 = nn.Conv2d(ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if args.arch == 'resnet34':
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, classes)

    # vgg16
    elif args.arch == 'vgg16':
        # model.features[0] = nn.Conv2d(ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=classes, bias=True)

    # densenet121
    elif args.arch == 'densenet121':
        # model.features.conv0 = nn.Conv2d(ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = nn.Linear(in_features=1024, out_features=classes, bias=True)

    elif args.arch == 'resnext50_32x4d':
        fc_features = model.fc.in_features
        model.fc = nn.Linear(in_features=fc_features, out_features=classes, bias=True)

    print(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:  # 分布式训练
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)  # 设置可用GPU
            model.cuda(args.gpu)
            # 在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，而是需要在程序中显示指定。
            # 调用model.cuda()，可以将模型加载到GPU上去。这种方法不被提倡，而建议使用model.to(device)
            # 的方式，这样可以显示指定需要使用的计算资源，特别是有多个GPU的情况下

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            # workers将负责的batch加载进RAM
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # 创建并行模型
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:  # pipeline训练
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:  # DataParallel
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)  # 数据通信，损失计算

    # SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Adam optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)

    # optionally resume from a checkpoint
    if args.resume:  # checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'Train_classification_112_80p')
    valdir = os.path.join(args.data, 'Test_classification_112_80p')

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    train_dataset = FUSARMapV2(traindir)
    val_dataset = FUSARMapV2(valdir)
    # normalize = transforms.Normalize(mean=[127.5],
    #                                  std=[127.5])
    #
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         # transforms.RandomResizedCrop(224),
    #         transforms.Resize(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    if args.distributed:
        # Sampler对应Worker
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(224),
    #         # transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    if args.evaluate:  # test
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        top1_acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
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

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
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

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg


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
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.1 ** (epoch // 10))
    # lr = args.lr * (0.5 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# lr scheduler for training
# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate based on schedule"""
#     lr = args.lr
#     if args.cos:  # cosine lr schedule
#         lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
#     else:  # stepwise lr schedule
#         for milestone in args.schedule:
#             lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
