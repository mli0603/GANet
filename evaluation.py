from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
import sys
import shutil
import os
import re
from struct import unpack
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.GANet_deep import GANet
from torch.utils.data import DataLoader
from dataloader.dataset import DatasetFromList
# from dataloader.data import get_test_set
import numpy as np
from natsort import natsorted

parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--dataset', type=str, default='sceneflow', help='sceneflow, kitti2015')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")
parser.add_argument('--multi_gpu', type=int, default=0, help="multi_gpu choice")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")

opt = parser.parse_args()

print(opt)

cuda = opt.cuda
# cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# print('===> Loading datasets')
# test_set = get_test_set(opt.data_path, opt.test_list, [opt.crop_height, opt.crop_width], false, opt.kitti, opt.kitti2015)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = GANet(opt.max_disp)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def test(input1, input2):
    _, _, height, width = input1.size()
    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[:, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[:, :, :]
    # skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    return temp


if __name__ == "__main__":
    if opt.dataset == 'sceneflow':
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
        right_data = [left.replace('left', 'right') for left in left_data]
        disp_data = [left.replace('frame_finalpass', 'disparity').replace('.png', '.pfm') for left in left_data]

        directory = os.path.join(file_path, 'occlusion', 'TEST', 'left')
        occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
        occ_data = natsorted(occ_data)
    elif opt.dataset == 'kitti2015':
        file_path = opt.data_path
        left_data = natsorted(
            [os.path.join(file_path, 'image_2/', img) for img in os.listdir(os.path.join(file_path, 'image_2/')) if
             img.find('_10') > -1])
        right_data = [img.replace('image_2/', 'image_3/') for img in left_data]
        disp_data = [img.replace('image_2/', 'disp_noc_0/') for img in left_data]
        occ_data = None

    test_data = DatasetFromList(left_data, right_data, disp_data, occ_data, opt.crop_height, opt.crop_width)
    test_dataloader = DataLoader(test_data, batch_size=3, shuffle=False, num_workers=2)

    avg_error = 0.0
    avg_wrong = 0.0
    avg_total = 0.0
    for index, (left, right, disp) in enumerate(test_dataloader):
        prediction = test(left, right)
        disp = disp.squeeze(1).cuda()  # NxHxW
        mask = disp > 0.0

        error = F.l1_loss(prediction[mask], disp[mask], reduction='none')
        wrong = torch.sum(error > opt.threshold).item()
        total = error.numel()
        avg_error = avg_error + torch.mean(error)
        avg_wrong += wrong
        avg_total += total
        print("===> Iteration {}: ".format(index) + " ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(
            torch.mean(error), wrong / total))

        # np.save('left.npy', left.data.cpu())
        # np.save('right.npy', right.data.cpu())
        # np.save('disp_pred.npy', prediction)
        # np.save('disp.npy', disp)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(disp[0])
        # plt.figure()
        # plt.imshow(prediction[0])
        # plt.show()
        # asdf

    avg_error = avg_error / len(test_dataloader)
    avg_rate = avg_wrong / avg_total
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Error Rate: {:.4f}".format(len(test_dataloader),
                                                                                          avg_error, avg_rate))
