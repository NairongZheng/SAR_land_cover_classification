"""
    data:2021.12.1
"""
import os
import argparse
import collections
import cv2
import numpy as np
import time
import math
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models

Image.MAX_IMAGE_PIXELS = None           # 防溢出

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def NMT(img, img_mean, rate=2.5, EPS=1e-7):
    img[img > (rate * img_mean + EPS)] = rate * img_mean
    img = (img / (np.max(img) + EPS)) * 255.
    img = img.astype('uint8')

    return img

def make_datasets(img_path, height=112, width=112, stride=14):
    # [h, w, s] = [112, 112, 56]
    # [h, w, s] = [112, 112, 28]
    # stride is the classification window
    img = Image.open(img_path)
    img = np.array(img, dtype='float32')
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # color - 4
    # img = img.astype(np.float32)[:, :, ::-1]  # bgr -> rgb

    # NMT preprocessing
    img_mean = 0.308
    img = NMT(img, img_mean)
    img = img.astype(np.float32)[:, :, np.newaxis]
    img = np.tile(img, (1, 1, 3))

    Height = img.shape[0]
    Width = img.shape[1]
    # ch = img.shape[2]

    n_row = math.floor((Height - height) / stride) + 1
    n_col = math.floor((Width - width) / stride) + 1

    samples = np.zeros((n_row * n_col, height, width, 3), dtype=np.uint8)

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

class ClassificationDatasetLoader():
    def __init__(self, root, transform=True):
        super(ClassificationDatasetLoader, self).__init__()

        self.root = root
        self.transform = transform
        self.samples, self.n_row, self.n_col, self.Height, self.Width = self.make_dataset(self.root)
    
    def make_dataset(self, img_path):
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

    def __getitem__(self, index):
        sample = self.samples[index]
        if not self.transform:
            sample = self.input_transform(sample)
            sample = sample.transpose((2, 0, 1))  # [c, h, w]
        else:
            sample = self.gen_samples(sample)

        return sample.copy(), self.n_row, self.n_col, self.Height, self.Width

    def __len__(self):
        return self.samples.shape[0]
    
    def inference(self, model, image):
        pred = model(image)
        return pred  # .exp()
    
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

def parse_args(model_names):
    """
        initialize parameter
    """
    parser = argparse.ArgumentParser(description='test classification network')

    # 测试的时候，直接给大图
    parser.add_argument('--data', help='path to test dataset', 
                    default='/emwuser/znr/code/micro_wave/single_pol_sar/test',
                    type=str, metavar='DIR')

    # resnet34, vgg16, densener121, resnext50_32x4d
    parser.add_argument('--arch', help='model architecture: ' + ' | '.join(model_names), default='densenet121',
                    metavar='ARCH', choices=model_names)
    parser.add_argument('--batch_size', help='mini-batch size', default=8, type=int)
    parser.add_argument('--classes', help='the number of classes', default=10, type=int)
    parser.add_argument('--pretrained', help='the test model dir', default='/emwuser/znr/code/micro_wave/classification/code', type=str)
    parser.add_argument('--weights_name', help='the test model name', default='densenet121_epoch23_TrainAcc98.90_ValAcc72.30_ckpt.pth.tar', type=str)
    parser.add_argument('--svd', help='the output path of test images', default='/emwuser/znr/code/micro_wave/classification/test_output', type=str)
    parser.add_argument('--seed', help='seed for initializing training', default=None, type=int)
    parser.add_argument('--gpu', help='GPU id to use', default=0, type=int)
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
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu
    begin_time = time.time()
    # create model
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
    

    model_state_file = os.path.join(args.pretrained, args.weights_name)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()

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
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('load model pre-trained done!')

    cudnn.benchmark = True

    data_dir = args.data
    f_names, names = create_filename(data_dir)
    for i in range(len(names)):
        f_name = f_names[i]
        print(f_name)
        # 文件加载，文件名循环，改变保存的文件名
        test_dataset = ClassificationDatasetLoader(f_name)

        # batch_size
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=args.batch_size, shuffle=False,
                                                    num_workers=0, pin_memory=True)

        name = names[i]
        # 增加batch_size不为1的测试算法，确保能还原
        test(test_dataset, testloader, model, name, args)
    

    end_time = time.time()
    time_used = end_time - begin_time
    print('Inference time : {}s'.format(time_used))

def test(test_dataset, testloader, model, name, args):
    model.eval()
    gpu = args.gpu
    n_pred = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(testloader)):
            image, n_row, n_col, Height, Width = batch
            image = image.cuda(gpu)  # non_blocking=True
            pred = test_dataset.inference(model, image)
            pred = pred.cpu().numpy()  # [B, H, W]  # (batch_size, classes)
            pred = np.argmax(pred, axis=1)          # (batch_size, )
            pred = list(pred)  # [1, 2, ..., batch], B x C
            n_pred += pred  # extend
        res_np = np.asarray(n_pred, dtype='uint8')
        res_np = np.reshape(res_np, (int(n_row.numpy()[0]), int(n_col.numpy()[0])))

        sv_path = args.svd
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        test_dataset.save_pred_10(res_np, int(Height.numpy()[0]), int(Width.numpy()[0]), name, sv_path)

if __name__ == '__main__':
    main()
