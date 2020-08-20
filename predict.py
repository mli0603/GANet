from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
# from GCNet.modules.GCNet import L1Loss
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from models.GANet_deep import GANet
from dataloader.data import get_test_set
import numpy as np
from natsort import natsorted
from dataloader.dataset import readPFM

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")

opt = parser.parse_args()

print(opt)
if opt.model == 'GANet11':
    from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from models.GANet_deep import GANet
else:
    raise Exception("No suitable model found ...")

cuda = opt.cuda
# cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# torch.manual_seed(opt.seed)
# if cuda:
#    torch.cuda.manual_seed(opt.seed)
# print('===> Loading datasets')


print('===> Building model')
model = GANet(opt.max_disp)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def test_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w


def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data


def test(leftname, rightname, dispname, occname, savename=''):
    #  count=0

    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    disp, h, w = readPFM(dispname)
    occ = np.array(Image.open(occname)).astype(np.bool)
    if h > opt.crop_height or w > opt.crop_width:
        start_x = int((w - opt.crop_width) / 2)
        start_y = int((h - opt.crop_height) / 2)
        disp = disp[start_y: start_y + opt.crop_height, start_x: start_x + opt.crop_width]
        occ = occ[start_y: start_y + opt.crop_height, start_x: start_x + opt.crop_width]
    disp[occ] = 0.0

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    # skimage.io.imsave(savename, (temp * 256).astype('uint16'))

    np.save('left.npy', input1.data.cpu())
    np.save('right.npy', input2.data.cpu())
    np.save('disp_pred.npy', temp)
    np.save('disp.npy', disp)

    print(np.abs(temp[~occ] - disp[~occ]).mean())


if __name__ == "__main__":
    file_path = opt.data_path
    directory = os.path.join(file_path, 'frame_finalpass', 'TEST')
    sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                   os.path.isdir(os.path.join(directory, subset))]

    seq_folders = []
    for sub_folder in sub_folders:
        seq_folders += [os.path.join(sub_folder, seq) for seq in os.listdir(sub_folder) if
                        os.path.isdir(os.path.join(sub_folder, seq))]

    left_data = []
    for seq_folder in seq_folders:
        left_data += [os.path.join(seq_folder, 'left', img) for img in
                      os.listdir(os.path.join(seq_folder, 'left'))]

    left_data = natsorted(left_data)

    directory = os.path.join(file_path, 'occlusion', 'TEST', 'left')
    occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
    occ_data = natsorted(occ_data)

    for index in range(len(left_data)):
        leftname = left_data[index]
        rightname = leftname.replace('left', 'right')
        disp_name = leftname.replace('frame_finalpass', 'disparity').replace('.png', '.pfm')
        occ_name = occ_data[index]

        # savename = opt.save_path + current_file[0: len(current_file) - 1]
        test(leftname, rightname, disp_name, occ_name)
