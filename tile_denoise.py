import torch
import torch.nn as nn
from utile import readtiff, writetiff
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from utile import readtiff, similarity, arr_log_normalization

from torch.utils.data import DataLoader

from torchvision.utils import save_image
import math


def create_tile(image,tilesize, stride=1):
    x = image.shape[1]
    step_num = math.ceil((x-tilesize+1)/stride)+1
    input_batch = np.empty((step_num**2,1,tilesize,tilesize))
    print(x-tilesize+1)
    for i in range(step_num):
        for j in range(step_num):
            if i == j & i == step_num-1:
                input_batch[step_num * i + j] = image[0, -tilesize:, -tilesize:]
            elif i == step_num-1:
                input_batch[step_num*i+j] = image[0,-tilesize:, j*stride:j*stride+tilesize]
            elif j ==  step_num-1:
                input_batch[step_num * i + j] = image[0, i*stride:i*stride+tilesize:, -tilesize:]
            else:
                input_batch[step_num*i+j] = image[0,i*stride:i*stride+tilesize, j*stride:j*stride+tilesize]

    return input_batch

def merge_tile(output,img_size, stride=1):
    outfeature = np.zeros((1,img_size, img_size))
    stack_num = np.zeros((1,img_size, img_size))
    tile_size = output.shape[2]
    step_num = math.ceil((img_size - tile_size + 1) / stride) + 1
    for i in range(step_num):
        for j in range(step_num):

            if i == j & i == step_num-1:
                outfeature[:, -tile_size:, -tile_size:] += output[step_num * i + j]
                stack_num[:, -tile_size:, -tile_size:] += 1
            elif i == step_num-1:
                outfeature[:, -tile_size:, j*stride:j*stride+tile_size] += output[step_num * i + j]
                stack_num[:, -tile_size:, j*stride:j*stride+tile_size] += 1
            elif j == step_num-1:
                outfeature[:, i*stride:i*stride+tile_size:, -tile_size:] += output[step_num * i + j]
                stack_num[:, i*stride:i*stride+tile_size:, -tile_size:] += 1
            else:
                outfeature[:,i*stride:i*stride+tile_size,j*stride:j*stride+tile_size] += output[step_num*i+j]
                stack_num [:,i*stride:i*stride+tile_size,j*stride:j*stride+tile_size] += 1

    print(stack_num)
    outimage = outfeature/stack_num
    print(outimage.shape)
    return outimage

def create_tile_cutedge(image, tilesize, cutsize=4):
    input_batch = create_tile(image, tilesize+2*cutsize,tilesize)
    return input_batch

def merge_tile_cutedge(output, img_size, cutsize=4):
    output = output[:,:,cutsize:-cutsize,cutsize:-cutsize]
    print(output.shape)
    print(img_size)
    print(output.shape[2])
    return merge_tile(output, img_size, output.shape[2])

class DenoiseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=48):
        super(DenoiseNet, self).__init__()
        self.upinput1 = nn.ConvTranspose2d(1,64,4,stride=4)
        self.conv0 = nn.Conv2d(64, 128, 3, padding=1, padding_mode="reflect")
        self.conv1 = nn.Conv2d(128, 256, 3, padding=1, padding_mode="reflect")
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect")
        self.pool2 = nn.MaxPool2d(2)


        self.bottleneck = nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect")


        self.upsamp2 = nn.Upsample(scale_factor=2)
        self.upconv2 = DenoiseNet._upblock(512, 256, "up2")
        self.upsamp1 = nn.Upsample(scale_factor=2)
        self.upconv1 = DenoiseNet._upblock(256 + 64, 96, "up1")

        self.convfin1 = nn.Conv2d(96, 32, 3, padding=1, padding_mode="reflect")
        self.convfin2 = nn.Conv2d(32, 1, 3, padding=1, padding_mode="reflect")

        self.downoutput = nn.Conv2d(1,1,4,stride=4, padding=0) #nn.AvgPool2d(4)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x=self.upinput1(x)
        encode1 = self.conv0(x)
        encode1 = self.act(encode1)
        encode1 = self.conv1(encode1)
        encode1 = self.act(encode1)
        pool1 = self.pool1(encode1)
        encode2 = self.conv2(pool1)
        encode2 = self.act(encode2)
        pool2 = self.pool2(encode2)


        bottleneck = self.bottleneck(pool2)


        cat2 = torch.cat((self.upsamp2(bottleneck), pool1), dim=1)
        decode2 = self.upconv2(cat2)
        cat1 = torch.cat((self.upsamp1(decode2), x), dim=1)
        decode1 = self.upconv1(cat1)

        x = self.act(self.convfin1(decode1))
        x = self.act(self.convfin2(x))
        x = self.downoutput(x)
        return x

    @staticmethod
    def _upblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",
                     nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, bias=False, padding=1,
                               padding_mode="reflect"),),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (name + "conv2",
                     nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, bias=False, padding=1,
                               padding_mode="reflect"),),
                    (name + "relu2", nn.LeakyReLU(inplace=True))
                ]
            )
        )

class DenoiseNet_transpose(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=48):
        super(DenoiseNet_transpose, self).__init__()
        self.upinput1 = nn.ConvTranspose2d(1,64,4,stride=4)
        self.conv0 = nn.Conv2d(64, 128, 3, padding=1, padding_mode="reflect")
        self.conv1 = nn.Conv2d(128, 256, 3, padding=1, padding_mode="reflect")
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect")
        self.pool2 = nn.MaxPool2d(2)


        self.bottleneck = nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect")


        #self.upsamp2 = nn.Upsample(scale_factor=2)
        self.upsamp2 = nn.ConvTranspose2d(256,256,2,stride=2)
        self.upconv2 = DenoiseNet._upblock(512, 256, "up2")
        #self.upsamp1 = nn.Upsample(scale_factor=2)
        self.upsamp1 = nn.ConvTranspose2d(256,256,2,stride=2)
        self.upconv1 = DenoiseNet._upblock(256 + 64, 96, "up1")

        self.convfin1 = nn.Conv2d(96, 32, 3, padding=1, padding_mode="reflect")
        self.convfin2 = nn.Conv2d(32, 1, 3, padding=1, padding_mode="reflect")

        self.downoutput = nn.Conv2d(1,1,4,stride=4, padding=0) #nn.AvgPool2d(4)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x=self.upinput1(x)
        encode1 = self.conv0(x)
        encode1 = self.act(encode1)
        encode1 = self.conv1(encode1)
        encode1 = self.act(encode1)
        pool1 = self.pool1(encode1)
        encode2 = self.conv2(pool1)
        encode2 = self.act(encode2)
        pool2 = self.pool2(encode2)


        bottleneck = self.bottleneck(pool2)


        cat2 = torch.cat((self.upsamp2(bottleneck), pool1), dim=1)
        decode2 = self.upconv2(cat2)
        cat1 = torch.cat((self.upsamp1(decode2), x), dim=1)
        decode1 = self.upconv1(cat1)

        x = self.act(self.convfin1(decode1))
        x = self.act(self.convfin2(x))
        x = self.downoutput(x)
        return x

    @staticmethod
    def _upblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",
                     nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, bias=False, padding=1,
                               padding_mode="reflect"),),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (name + "conv2",
                     nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, bias=False, padding=1,
                               padding_mode="reflect"),),
                    (name + "relu2", nn.LeakyReLU(inplace=True))
                ]
            )
        )

class DenoiseNet_transpose_modified(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=48):
        super(DenoiseNet_transpose_modified, self).__init__()
        self.upinput1 = nn.ConvTranspose2d(1,64,4,stride=4)
        self.conv0 = nn.Conv2d(64, 128, 3, padding=1, padding_mode="reflect")
        self.conv1 = nn.Conv2d(128, 256, 3, padding=1, padding_mode="reflect")
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect")
        self.pool2 = nn.MaxPool2d(2)


        self.bottleneck = nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect")


        #self.upsamp2 = nn.Upsample(scale_factor=2)
        self.upsamp2 = nn.ConvTranspose2d(256,256,2,stride=2)
        self.upconv2 = DenoiseNet._upblock(512, 256, "up2")
        #self.upsamp1 = nn.Upsample(scale_factor=2)
        self.upsamp1 = nn.ConvTranspose2d(256,256,2,stride=2)
        self.upconv1 = DenoiseNet._upblock(256 + 64, 96, "up1")

        self.convfin1 = nn.Conv2d(96, 32, 3, padding=1, padding_mode="reflect")
        self.convfin2 = nn.Conv2d(32, 1, 3, padding=1, padding_mode="reflect")

        self.downoutput = nn.Conv2d(1,1,4,stride=4, padding=0) #nn.AvgPool2d(4)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x=self.act(self.upinput1(x))
        encode1 = self.conv0(x)
        encode1 = self.act(encode1)
        encode1 = self.conv1(encode1)
        encode1 = self.act(encode1)
        pool1 = self.pool1(encode1)
        encode2 = self.conv2(pool1)
        encode2 = self.act(encode2)
        pool2 = self.pool2(encode2)


        bottleneck = self.bottleneck(pool2)


        cat2 = torch.cat((self.upsamp2(bottleneck), pool1), dim=1)
        decode2 = self.upconv2(cat2)
        cat1 = torch.cat((self.upsamp1(decode2), x), dim=1)
        decode1 = self.upconv1(cat1)

        x = self.act(self.convfin1(decode1))
        x = self.act(self.convfin2(x))
        x = self.downoutput(x)
        return x

    @staticmethod
    def _upblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",
                     nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, bias=False, padding=1,
                               padding_mode="reflect"),),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (name + "conv2",
                     nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, bias=False, padding=1,
                               padding_mode="reflect"),),
                    (name + "relu2", nn.LeakyReLU(inplace=True))
                ]
            )
        )

class DenoiseNet_transpose_modified2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=48):
        super(DenoiseNet_transpose_modified2, self).__init__()
        self.upinput1 = nn.ConvTranspose2d(1,64,4,stride=4)
        self.conv0 = nn.Conv2d(64, 128, 3, padding=1, padding_mode="reflect")
        self.conv1 = nn.Conv2d(128, 256, 3, padding=1, padding_mode="reflect")
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect")
        self.pool2 = nn.MaxPool2d(2)


        self.bottleneck = nn.Conv2d(256, 256, 3, padding=1, padding_mode="reflect")


        #self.upsamp2 = nn.Upsample(scale_factor=2)
        self.upsamp2 = nn.ConvTranspose2d(256,256,2,stride=2)
        self.upconv2 = DenoiseNet._upblock(512, 256, "up2")
        #self.upsamp1 = nn.Upsample(scale_factor=2)
        self.upsamp1 = nn.ConvTranspose2d(256,256,2,stride=2)
        self.upconv1 = DenoiseNet._upblock(256 + 64, 96, "up1")

        self.convfin1 = nn.Conv2d(96, 32, 3, padding=1, padding_mode="reflect")
        self.convfin2 = nn.Conv2d(32, 1, 3, padding=1, padding_mode="reflect")

        self.downoutput = nn.Conv2d(1,1,4,stride=4, padding=0) #nn.AvgPool2d(4)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x=self.act(self.upinput1(x))
        encode1 = self.conv0(x)
        encode1 = self.act(encode1)
        encode1 = self.conv1(encode1)
        encode1 = self.act(encode1)
        pool1 = self.pool1(encode1)
        encode2 = self.conv2(pool1)
        encode2 = self.act(encode2)
        pool2 = self.pool2(encode2)
        bottleneck = self.bottleneck(pool2)
        cat2 = torch.cat((self.act(self.upsamp2(bottleneck)), pool1), dim=1)
        decode2 = self.upconv2(cat2)
        cat1 = torch.cat((self.act(self.upsamp1(decode2)), x), dim=1)
        decode1 = self.upconv1(cat1)

        x = self.act(self.convfin1(decode1))
        x = self.act(self.convfin2(x))
        x = self.downoutput(x)
        return x

    @staticmethod
    def _upblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",
                     nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, bias=False, padding=1,
                               padding_mode="reflect"),),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (name + "conv2",
                     nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, bias=False, padding=1,
                               padding_mode="reflect"),),
                    (name + "relu2", nn.LeakyReLU(inplace=True))
                ]
            )
        )


PATH = './models/test/senti_multi_step2_00_05_modified2_1.pth'
net = DenoiseNet_transpose_modified2()
net.eval()
net = nn.DataParallel(net)
net.to('cuda')
net.module.load_state_dict(torch.load(PATH))

brightness_t1 = readtiff('data/sentinental_1/time3.tif')
brightness_t1[brightness_t1==0] = np.min(brightness_t1[np.nonzero(brightness_t1)])
brightness_t1 = arr_log_normalization(brightness_t1)*2
#brightness_t1 = brightness_t1[:512,-512:]
brightness_t1 = np.pad(brightness_t1,4,'reflect')

input = np.expand_dims(brightness_t1, 0)

#input = create_tile(input, 64,2)
input = create_tile_cutedge(input, 64, 4)
input = input.astype('float32')
input = torch.tensor(input)
print(input.shape)


net.to('cuda')

input_dl = DataLoader(input, batch_size=10, shuffle=False)
outputs_list = []
count = np.ceil(input.shape[0]/input_dl.batch_size)
with torch.no_grad():
    for input in input_dl:
        input = input.to('cuda')
        outputs = net(input)
        outputs = outputs.to('cpu').numpy()

        outputs = np.exp(outputs/2)


        outputs_list.append(outputs)
        count = count - 1
        print(count)
outputs_batch = np.concatenate(outputs_list,axis=0)
print(outputs_batch.shape)
#outimage = merge_tile(outputs_batch,brightness_t1.shape[1],2)
print(brightness_t1.shape[1])
outimage = merge_tile_cutedge(outputs_batch,brightness_t1.shape[1]-8,4)
outimage = np.squeeze(outimage,axis=0).astype('float32')
print(outimage)

writetiff(outimage, "./output/test/senti_time3_single_step2_siamese_00_05_transpose_modified2.tif")