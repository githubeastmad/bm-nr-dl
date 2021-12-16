import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import random
from collections import OrderedDict


import numpy as np
import os

import torch.optim as optim



from utile import readtiff, writetiff, similarity,  arr_log_normalization, point_feature

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import math
import multiprocessing as mp
from multiprocessing import Process


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


        bottleneck = self.act(self.bottleneck(pool2))


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


def sim_loss(outputs, labels):
    print(outputs.shape)
    print(labels.shape)

def sim_loss(output, target, sim):
    output = output.clone()
    output = output - target
    output = torch.mul(sim,output)
    output = output.pow_(2)
    output = output.mean()

    return output

def nosim_loss(output, target):
    output = torch.exp(output)
    target = torch.exp(target)
    output = torch.log((output + target) / torch.sqrt(output * target))

    output = torch.sum(output)
    return output

def prior_loss(output,target):
    output = torch.exp(output)
    target = torch.exp(target)
    output = torch.sum(((output - target)**2) / (output * target))
    return output

def divide_chunks(l,n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def nlm(image, featureimage, m=7, search=21, smooth=2):
    feature = point_feature(featureimage, m)
    print(feature.shape)

    image_pad = np.pad(image, int(math.ceil(search - 1) / 2), 'symmetric')
    feature_pad = np.pad(feature, ((0,0), (int(math.ceil(search - 1) / 2), int(math.ceil(search - 1) / 2)),(int(math.ceil(search - 1) / 2), int(math.ceil(search - 1) / 2)))
                         , 'empty')

    image_cal = np.empty((search**2, image.shape[0], image.shape[1]), dtype='float64')
    feature_cal = np.empty((search**2, image.shape[0], image.shape[1]),dtype='float64')


    for i in range(search):
        for j in range(search):
            feature2 = feature_pad[:, i:i - search + 1 or None, j:j - search + 1 or None]
            l2_sim = np.exp(-(1 / smooth**2) * np.sqrt(np.sum(np.square(feature2 - feature), axis=0)))

            image_cal[i*search+j] = image_pad[i:i - search + 1 or None, j:j - search + 1 or None]
            feature_cal[i*search+j] = l2_sim
    image_cal = np.delete(image_cal, int((search**2-1)/2),axis=0)
    feature_cal = np.delete(feature_cal, int((search ** 2 - 1) / 2), axis=0)
    output = np.sum(image_cal * feature_cal / np.sum(feature_cal,axis=0),axis=0)
    return output

def searchsimilar(inputsearchfeature):
    inputfeature = inputsearchfeature[0]
    searchfeature = inputsearchfeature[1]


    k = inputsearchfeature[2]

    inputfeature = np.exp(inputfeature)
    searchfeature = np.exp(searchfeature)


    input_origin_feature = np.split(inputfeature, 2,axis=0)[0]
    input_denoised_feature = np.split(inputfeature, 2, axis=0)[1]
    input_denoised_feature = np.sqrt(input_denoised_feature)

    mindist = [(np.inf,None)]*k

    #minfeature = np.empty((searchfeature.shape[0],k))

    for i in range(searchfeature.shape[1]):
        for j in range(searchfeature.shape[2]):
            comparefeature = searchfeature[:, i, j]

            compare_origin_feature = np.split(comparefeature, 2)[0]
            compare_denoised_feature = np.split(comparefeature, 2)[1]


            if np.array_equal(inputfeature, comparefeature):
                continue


            compare_denoised_feature = np.sqrt(compare_denoised_feature)
            dist = np.sum(np.log(np.sqrt(np.divide(input_origin_feature, compare_origin_feature)) + np.sqrt(np.divide(compare_origin_feature, input_origin_feature)))
                          + 2.0*np.square(input_denoised_feature - compare_denoised_feature)/np.multiply(input_denoised_feature,compare_denoised_feature))

            if mindist[k-1][0] > dist:
                mindist[k-1] = (dist,np.log(compare_origin_feature))
                #minfeature[k-1] =
                mindist.sort(key=lambda x: x[0], reverse=False)

    inputfeature = np.log(input_origin_feature)



    return inputfeature, mindist

def set_affinity_on_worker():
    os.system("taskset -cp 0-%d %s" % (mp.cpu_count(),os.getpid()))

if __name__ == '__main__':
    os.system("taskset -p %s" %os.getpid())
    os.system("taskset -cp 0-%d %s" % (mp.cpu_count(),os.getpid()))
    os.system("taskset -p %s" %os.getpid())
    pool = mp.Pool(mp.cpu_count())

    SEARCHRADIUS = 100
    KERNALSIZE = 13
    PAIRNUM = 40000
    PROCESS = 12
    N=16
    THRESHOLD = 1.26
    TRAINEPOCH = 7
    brightness_t1 = readtiff('data/terrasar/time1.tif')
    brightness_t1[brightness_t1 == 0] = np.min(brightness_t1[np.nonzero(brightness_t1)])
    brightness_t1 = arr_log_normalization(brightness_t1)*2
    brightness_t1_1 = readtiff('./output/test/terrasar_time1_single_step1_siamese_00_transpose_modified2.tif')
    brightness_t1_1[brightness_t1_1 == 0] = np.min(brightness_t1_1[np.nonzero(brightness_t1_1)])
    brightness_t1_1 = arr_log_normalization(brightness_t1_1)*2

    print(brightness_t1.shape)
    print(brightness_t1_1.shape)


    feature_t1 = point_feature(brightness_t1,KERNALSIZE).astype('float32')

    feature_t1_1 = point_feature(brightness_t1_1, KERNALSIZE).astype('float32')

    feature_t1 = np.concatenate((feature_t1, feature_t1_1))



    train_batch = np.empty((PAIRNUM * N, 1, KERNALSIZE, KERNALSIZE))
    target_batch = np.empty((PAIRNUM * N, 1, KERNALSIZE, KERNALSIZE))
    del_index = []

    input_search_batch = [None]*PAIRNUM


    # creating training tiles
    for i in range(PAIRNUM):
        if i%1000 == 0:
            print(str(i))
        x = random.randrange(SEARCHRADIUS, feature_t1.shape[1]-SEARCHRADIUS, 3)
        y = random.randrange(SEARCHRADIUS, feature_t1.shape[2]-SEARCHRADIUS, 3)


        input_img_origin = brightness_t1[x - KERNALSIZE:x, y - KERNALSIZE:y]
        input_img_denoised = brightness_t1_1[x - KERNALSIZE:x, y - KERNALSIZE:y]
        input_feature = np.concatenate((input_img_origin.flatten(),input_img_denoised.flatten()))
        #search = brightness_t1[x-SEARCHRADIUS:x+SEARCHRADIUS, y-SEARCHRADIUS:y+SEARCHRADIUS]
        search_feature = feature_t1[:,x-SEARCHRADIUS:x+SEARCHRADIUS, y-SEARCHRADIUS:y+SEARCHRADIUS]
        #search_feature = search_feature.reshape((KERNALSIZE**2,search_feature.shape[1]*search_feature.shape[2]))



        input_search_batch[i] = (input_feature, search_feature, N)



    similarpatchs = pool.imap_unordered(searchsimilar, input_search_batch)
    similarpatchs = [similarpatch for similarpatch in similarpatchs]

    for i, similarpatch in enumerate(similarpatchs):
        for j,similar in enumerate(similarpatch[1]):
            if similar[0]/KERNALSIZE**2<THRESHOLD:
                train_batch[i * N + j, 0] = similarpatch[0].reshape((KERNALSIZE, KERNALSIZE))
                target_batch[i * N + j, 0] = similar[1].reshape((KERNALSIZE, KERNALSIZE))
            else:
                del_index.append(i * N + j)


    print(target_batch.shape)
    train_batch = np.delete(train_batch, del_index, 0)
    target_batch = np.delete(target_batch, del_index, 0)
    print(train_batch.shape)
    pool.close()
    #input("samples ready")
    train_batch = train_batch.astype('float32')
    target_batch = target_batch.astype('float32')

    #train_batch, target_batch = np.concatenate((train_batch, target_batch)), np.concatenate((target_batch, train_batch))
    print(train_batch.shape)
    print(target_batch.shape)

    #print(sim_batch.shape)


    train_ds = TensorDataset(torch.tensor(train_batch), torch.tensor(target_batch))

    train_dl = DataLoader(train_ds, batch_size=300, pin_memory=True, shuffle=False)

    net = DenoiseNet_transpose(1, 1, 48)

    net = nn.DataParallel(net)
    net.to("cuda")


    opt = optim.Adam(net.parameters(), betas=(0.9, 0.99), lr=0.0001)


    for epoch in range(TRAINEPOCH):
        print('current epoch:'+str(epoch))
        for xb,yb in train_dl:
            opt.zero_grad()
            xb = xb.to("cuda")
            yb = yb.to("cuda")
            pred1 = net(xb)
            pred2 = net(yb)
            loss1 = nn.functional.l1_loss(pred1, yb)
            loss2 = nn.functional.l1_loss(pred2, xb)
            dist = nn.functional.l1_loss(pred1, pred2)
            loss = loss1 + loss2 + 0.2 * dist
            loss.backward()
            opt.step()
            #print(loss.item())


    PATH = './models/test/terrasar_time1_single_step2_siamese_00_02_transpose_modified2_2.pth'
    torch.save(net.module.state_dict(), PATH)
