import argparse
import collections
import os
import random
import shutil
import time
import warnings
import math
from tqdm import tqdm

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
from torch.nn import functional as F

from PIL import Image
# import os
import os.path
import numpy as np
from torch.utils import data

classes = 10
ch = 3
POLSAR = False

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

"""
python main.py -a resnet50
"""
parser.add_argument('--data',
                    default='/data/shixianzheng/2021_research/SAR_classification/FUSAR-Map-V2/singlePolSAR/SAR images',
                    # '/data/shixianzheng/2021_research/SAR_segmentation/FUSAR-MapV2/singlePolSAR/SAR images'
                    metavar='DIR',
                    help='path to dataset')

# resnet34, vgg16, densener121
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet34)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--pretrained',
                    default='/data/shixianzheng/2021_research/SAR_classification/FUSAR-Map-V2/singlePolSAR/output_112_80p_densenet121',
                    # '/data/shixianzheng/2021_research/SAR_classification/xzshi_code21/ouput_weights/output_112_50p_vgg16',
                    type=str, metavar='PATH',
                    help='use pre-trained model path')

parser.add_argument('--svd',
                    default='/data/shixianzheng/2021_research/SAR_classification/FUSAR-Map-V2/singlePolSAR/output_112_80p_densenet121/output_results',
                    # '/data/shixianzheng/2021_research/SAR_classification/xzshi_code21/ouput_weights/output_112_50p_vgg16/output_results',
                    type=str, metavar='PATH',
                    help='use pre-trained model path')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu', default=3, type=int,
                    help='GPU id to use.')


def NMT(img, img_mean, rate=2.5, EPS=1e-7):
    img[img > (rate * img_mean + EPS)] = rate * img_mean
    img = (img / (np.max(img) + EPS)) * 255.
    img = img.astype('uint8')

    return img


def mean_img(img, threshold=0, EPS=1e-7):
    black_pixel = 0
    sum_pixel = 0
    img_sum = 0
    black = (img <= (threshold + EPS)) * 1.
    black_pixel += np.sum(black)
    img_sum += np.sum(img)
    sum_pixel += img.shape[0] * img.shape[1]
    img_mean = img_sum / (sum_pixel - black_pixel + 1)
    return img_mean


def make_datasets(img_path, height=112, width=112, stride=14):
    # [h, w, s] = [112, 112, 56]
    # [h, w, s] = [112, 112, 28]
    # stride is the classification window
    if not POLSAR:
        img = Image.open(img_path)
        img = np.array(img, dtype='float32')
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # color - 4
        # img = img.astype(np.float32)[:, :, ::-1]  # bgr -> rgb

        # NMT preprocessing
        img_mean = 0.308
        img = NMT(img, img_mean)
        img = img.astype(np.float32)[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    else:
        img_0 = Image.open(img_path[0])
        img_0 = np.array(img_0, dtype='float32')
        img_mean_0 = mean_img(img_0)  # HH
        img_0 = NMT(img_0, img_mean_0)

        img_1 = Image.open(img_path[1])
        img_1 = np.array(img_1, dtype='float32')
        img_mean_1 = mean_img(img_1)  # HV
        img_1 = NMT(img_1, img_mean_1)

        img_2 = Image.open(img_path[2])
        img_2 = np.array(img_2, dtype='float32')
        img_mean_2 = mean_img(img_2)  # VH
        img_2 = NMT(img_2, img_mean_2)

        img_12 = (img_1 + img_2) / 2

        img_3 = Image.open(img_path[3])
        img_3 = np.array(img_3, dtype='float32')
        img_mean_3 = mean_img(img_3)  # VV
        img_3 = NMT(img_3, img_mean_3)

        img = np.zeros((img_0.shape[0], img_0.shape[1], 3))
        img[:, :, 0] = img_0
        img[:, :, 1] = img_12
        img[:, :, 2] = img_3

    Height = img.shape[0]
    Width = img.shape[1]
    # ch = img.shape[2]

    n_row = math.floor((Height - height) / stride) + 1
    n_col = math.floor((Width - width) / stride) + 1

    samples = np.zeros((n_row * n_col, height, width, ch), dtype=np.uint8)

    K = 0
    for m in range(n_row):
        row_start = m * stride
        row_end = m * stride + height
        for n in range(n_col):
            col_start = n * stride
            col_end = n * stride + width
            img_mn = img[row_start:row_end, col_start:col_end]
            samples[K] = img_mn
            K += 1

    return samples.copy(), n_row, n_col, Height, Width


class FUSARMapV2:
    def __init__(self, root, transform=True):
        self.root = root
        self.transform = transform
        self.samples, self.n_row, self.n_col, self.Height, self.Width = self.make_datasets(self.root)

    @staticmethod
    def make_datasets(img_path):
        return make_datasets(img_path)

    def input_transform(self, image):
        # image = image.astype(np.float32)[:, :, ::-1]  # 通道逆序, why?
        image = image.astype(np.float32)
        # image = np.tile(image, (1, 1, 3))
        image = image / 127.5 - 1  # [-1, 1]
        return image

    def image_resize(self, image, new_w=224, new_h=224):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return image

    def gen_samples(self, image):
        image = self.image_resize(image)
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))
        return image

    def inference(self, model, image):
        pred = model(image)
        return pred  # .exp()

    def __getitem__(self, index):
        sample = self.samples[index]
        # sample = self.loader(path)
        # sample = cv2.imread(path, cv2.IMREAD_COLOR)
        if not self.transform:
            sample = self.input_transform(sample)
            sample = sample.transpose((2, 0, 1))  # [c, h, w]
        else:
            sample = self.gen_samples(sample)

        return sample.copy(), self.n_row, self.n_col, self.Height, self.Width

    def __len__(self):
        return self.samples.shape[0]

    def save_pred(self, preds, Height, Width, name, sv_path):

        baresoil_mask = preds == 0
        residential_mask = preds == 1
        road_mask = preds == 2
        vegetation_mask = preds == 3
        water_mask = preds == 4
        woodland = preds == 5

        rgb = np.zeros((preds.shape[0], preds.shape[1], 3))
        print(rgb.shape)
        rgb[:, :, 0] = baresoil_mask * 139 + residential_mask * 255 + road_mask * 83
        rgb[:, :, 1] = vegetation_mask * 255 + woodland * 139 + road_mask * 134
        rgb[:, :, 2] = water_mask * 255 + road_mask * 139

        save_img = Image.fromarray(np.uint8(rgb))
        save_img = save_img.resize((Width, Height), Image.NEAREST)
        save_img.save(os.path.join(sv_path, 'pred_' + name + '_s14.png'))

    def save_pred_10(self, preds, Height, Width, name, sv_path):

        baresoil_mask = preds == 0
        humanbuilt_mask = preds == 1
        industry_mask = preds == 2
        paddyland_mask = preds == 3
        plantingland_mask = preds == 4
        residential_mask = preds == 5
        road_mask = preds == 6
        vegetation_mask = preds == 7
        water_mask = preds == 8
        woodland = preds == 9

        rgb = np.zeros((preds.shape[0], preds.shape[1], 3))
        print(rgb.shape)
        rgb[:, :, 0] = baresoil_mask * 139 + residential_mask * 205 + road_mask * 83 + industry_mask * 255 + plantingland_mask * 139 + humanbuilt_mask * 189
        rgb[:, :, 1] = vegetation_mask * 255 + residential_mask * 173 + woodland * 139 + road_mask * 134 + plantingland_mask * 105 + humanbuilt_mask * 183 + paddyland_mask * 139
        rgb[:, :, 2] = water_mask * 255 + road_mask * 139 + plantingland_mask * 20 + humanbuilt_mask * 107 + paddyland_mask * 139

        save_img = Image.fromarray(np.uint8(rgb))
        save_img = save_img.resize((Width, Height), Image.NEAREST)
        save_img.save(os.path.join(sv_path, 'pred_' + name + '_s14.png'))


def test(test_dataset, testloader, model, name, args):
    model.eval()

    gpu = args.gpu
    n_pred = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            image, n_row, n_col, Height, Width = batch
            image = image.cuda(gpu)  # non_blocking=True
            pred = test_dataset.inference(model, image)
            pred = pred.cpu().numpy()  # [B, H, W]
            pred = np.argmax(pred, axis=1)
            pred = list(pred)  # [1, 2, ..., batch], B x C
            n_pred += pred  # extend
        res_np = np.asarray(n_pred, dtype='uint8')
        res_np = np.reshape(res_np, (int(n_row.numpy()[0]), int(n_col.numpy()[0])))

        sv_path = args.svd
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        test_dataset.save_pred_10(res_np, int(Height.numpy()[0]), int(Width.numpy()[0]), name, sv_path)


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
    main_worker(args.gpu, args)


def create_filename(input_dir):
    img_filename = []
    names = []
    path_list = os.listdir(input_dir)
    path_list.sort()
    for filename in path_list:
        char_name = filename.split('.')[0]
        names.append(char_name)
        file_path = os.path.join(input_dir, filename)
        img_filename.append(file_path)

    return img_filename, names


def main_worker(gpu, args):
    args.gpu = gpu

    begin_time = time.time()

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # resnet34
    if args.arch == 'resnet34':
        # model.conv1 = nn.Conv2d(ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, classes)

    # vgg16
    elif args.arch == 'vgg16':
        # model.classifier = nn.Sequential(
        #     nn.Linear(in_features=25088, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=4096, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=4096, out_features=6, bias=True)
        # )
        model.classifier[6] = nn.Linear(in_features=4096, out_features=classes, bias=True)

    # densenet121
    elif args.arch == 'densenet121':
        # model.features.conv0 = nn.Conv2d(ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = nn.Linear(in_features=1024, out_features=classes, bias=True)

    # print(model)

    model_state_file = args.pretrained
    weights_name = 'model_best.pth.tar'
    model_state_file = os.path.join(model_state_file, weights_name)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()

    # for k, v in model_dict.items():
    #     print('{}'.format(k))

    # for k, v in pretrained_dict['state_dict'].items():
    #     print("{}".format(k))

    # resnet
    if args.arch == 'resnet34':
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict['state_dict'].items()
                           if k[7:] in model_dict.keys()}

    # vgg16
    elif args.arch == 'vgg16':
        tmp_dict = collections.defaultdict(lambda: 0)
        for k, v in pretrained_dict['state_dict'].items():
            if k[:8] == 'features':
                k_d = k[:8] + '.' + k[16:]
                tmp_dict[k_d] = v
            else:
                tmp_dict[k] = v

        pretrained_dict = tmp_dict

    # densenet121
    elif args.arch == 'densenet121':
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict['state_dict'].items()
                           if k[7:] in model_dict.keys()}

    elif args.arch == 'resnext50_32x4d':
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict['state_dict'].items()
                           if k[7:] in model_dict.keys()}

    # for k, v in pretrained_dict.items():
    #     print('{}'.format(k))

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('load model pre-trained done!')

    cudnn.benchmark = True

    # begin inference
    if not POLSAR:
        data_dir = args.data
        f_names, names = create_filename(data_dir)
        for i in range(len(names)):  # len(names)
            # if i == 4 or i == 6:
            f_name = f_names[i]
            print(f_name)
            # 文件加载，文件名循环，改变保存的文件名
            test_dataset = FUSARMapV2(f_name)

            # batch_size
            testloader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers, pin_memory=True)

            name = names[i]
            # 增加batch_size不为1的测试算法，确保能还原
            test(test_dataset, testloader, model, name, args)

    else:
        data_dir = args.data
        f_names, names = create_filename(data_dir)
        # 文件加载，文件名循环，改变保存的文件名
        test_dataset = FUSARMapV2(f_names)
        # batch_size
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
        name = names[0]
        # 增加batch_size不为1的测试算法，确保能还原
        test(test_dataset, testloader, model, name, args)

    end_time = time.time()
    time_used = end_time - begin_time
    print('Inference time : {}s'.format(time_used))


if __name__ == '__main__':
    main()
