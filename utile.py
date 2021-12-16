
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat

import scipy.io
import torch
import math

def readtiff(filename):
    im = Image.open(filename)
    imarray=np.array(im)
    return imarray

def blockshaped(arr, nrows, ncols, nbuf, ncan):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    print(h // (nrows + 2 * nbuf))
    print(w // (ncols + 2 * nbuf))
    i = 0

    startx = nrows + 2 * nbuf
    starty = ncols + 2 * nbuf

    stridex = nrows - ncan
    stridey = ncols - ncan

    endx, endy = arr.shape
    steps_x_extra = 1 if (endx - startx) % stridex else 0
    steps_y_extra = 1 if (endy - starty) % stridex else 0

    steps_x = (endx - startx) // stridex + 1
    steps_y = (endy - starty) // stridey + 1

    matrix = np.zeros(((steps_x + steps_y_extra) * (steps_y + steps_y_extra), startx, starty))
    print(matrix.shape)

    k = 0
    i = 0
    while i < steps_x:
        j = 0
        while j < steps_y:
            matrix[k] = arr[(i * stridex):(startx + i * stridex), (j * stridey):(starty + j * stridey)]
            k += 1
            print(k)
            print(i)
            print(j)
            if (steps_y_extra == 1 and j == steps_y - 1):
                matrix[k] = arr[(i * stridex):(startx + i * stridex), (endy - starty):endy]
                k += 1
                print(k)
                print(i)
                print(j)
            j += 1
        if (steps_x_extra == 1 and i == steps_x - 1):
            j = 0
            while j < steps_y:
                matrix[k] = arr[(endx - startx):endx, (j * stridey):(starty + j * stridey)]
                k += 1
                print(k)
                print(i)
                print(j)
                if (steps_y_extra == 1 and j == steps_y - 1):
                    matrix[k] = arr[(endx - startx):endx, (endy - starty):endy]
                j += 1
        i += 1
    print(k)
    return matrix, steps_x_extra, steps_y_extra


def unblockshaped(arr, h, w, ncan, xextra, yextra):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    print(arr.shape)
    stepx = h // nrows
    stepy = w // ncols
    stridex = nrows - ncan
    stridey = ncols - ncan
    print(stridey)
    print(stepy)
    i = 0

    matrix = np.zeros((h, w))
    while i < stepx:
        j = 0
        while j < stepy:
            print(matrix[(i * stridex):(i * stridex + nrows), (j * stridey):(j * stridey + ncols)].shape)
            matrix[(i * stridex):(i * stridex + nrows), (j * stridey):(j * stridey + ncols)] = arr[
                                                                                               i * (stepy + yextra) + j,
                                                                                               :, :]
            j += 1
        if (yextra):
            matrix[(i * stridex):(i * stridex + nrows), (w - ncols):w] = arr[i * (stepy + yextra) + j, :, :]
        i += 1
    if (xextra):
        j = 0
        while j < stepy:
            matrix[(h - nrows):h, (j * stridey):(j * stridey + ncols)] = arr[i * (stepy + yextra) + j, :, :]
            j += 1
        if (yextra):
            matrix[(h - nrows):h, (w - ncols):w] = arr[i * (stepy + yextra) + j, :, :]
    return matrix


def customloss(y_true, y_pred):
    return y_pred


def readtiff(filename):
    im = Image.open(filename)
    imarray = np.array(im)
    return imarray
def writetiff(x_arr, filename):
    b = Image.fromarray(x_arr)
    b.save(filename)
    return

def untile(file1, simfile, modelfile, outfile):
    brightness = readtiff(file1)

    print(brightness)
    # img_input, xextra, yextra=blockshaped(brightness, 256, 256,1, 3)

    input_similarity = readtiff(simfile)

    # img_input, xextra, yextra=blockshaped(brightness, 40, 40,0, 0)

    img_input = np.expand_dims(np.expand_dims(brightness, axis=3), axis=0)
    input_similarity = np.expand_dims(np.expand_dims(input_similarity, axis=3), axis=0)

    # print(i)
    autoencoder = load_model(modelfile, custom_objects={'customloss': customloss})
    denoise_im = autoencoder.predict(img_input)
    print(denoise_im.shape)
    denoise_im = np.squeeze(np.squeeze(denoise_im, axis=3), axis=0)
    # denoise_im=unblockshaped(denoise_im, 1600, 1600,0, xextra, yextra)
    savepath = outfile
    b = Image.fromarray(denoise_im)
    b.save(savepath)
    return



##cn1: numpy array; cn2:numpy array
def similarity_l2(cn1, cn2):
    return np.exp(-np.mean(np.square(cn1 - cn2)))

def similarity_l2_pixel(cn1, cn2):
    return np.exp(-np.mean(np.square(cn1 - cn2), axis=0))

def similarity(arr1, arr2):
    x, y = arr1.shape

    i = 0
    similarity = 0

    while i < x:

        j = 0
        while j < y:
            similarity = similarity + 4 * arr1[i, j] * arr2[i, j] / np.square(arr1[i, j] + arr2[i, j])

            j += 1
        i += 1

    return (similarity / (x * y))

def exp_similarity(arr1, arr2):
    x, y = arr1.shape

    i = 0
    similarity = 0

    arr1 = np.exp(arr1)
    arr2 = np.exp(arr2)

    return np.mean(np.log((arr1 + arr2) / np.sqrt(arr1*arr2)))


def arr_normalization(arr):
    m = min(arr.max(),500)
    arr = arr/ m
    return arr, m

def arr_log_normalization(arr):
    arr = np.log(arr)
    return arr

def arr_znormalization(arr):
    STD = 0.378
    MEAN = 0.27886
    output = (arr-MEAN)/STD

    return output - output.min()

def arr_deznormalization(arr):
    STD = 0.378
    MEAN = 0.27886
    output = arr*STD + MEAN
    return output - output.min()

def mat2tif(mat, key, tif):
    x = loadmat(mat)
    x = x[key]
    writetiff(x,tif)

def PCA_eig(X,k, center=True, scale=False):
  n,p = X.size()
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  X_center =  torch.mm(H.double(), X.double())
  covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
  scaling =  torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).double()
  scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
  eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
  components = (eigenvectors[:, :k]).t()
  explained_variance = eigenvalues[:k, 0]
  return { 'X':X, 'k':k, 'components':components,
    'explained_variance':explained_variance }


def PCA_svd(X, k, center=True):
  n = X.size()[0]
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  X_center =  torch.mm(H.double(), X.double())
  u, s, v = torch.svd(X_center)
  components  = v[:k].t()
  explained_variance = torch.mul(s[:k], s[:k])/(n-1)
  return { 'X':X, 'k':k, 'components':components,
    'explained_variance':explained_variance }

def awgn(x, sigma=1):
    noise = np.random.normal(0, sigma, x.shape)
    return x + noise

def similarity_point_sar(brightness, brightness2, m=3, smooth=1):
    n = int(math.ceil(m - 1) / 2)

    brightness_cal = np.pad(brightness, n, 'edge')
    brightness2_cal = np.pad(brightness2, n, 'edge')
    feature1 = np.empty((m*m, brightness.shape[0], brightness.shape[1]))
    feature2 = np.empty((m*m, brightness.shape[0], brightness.shape[1]))



    for i in range(m):
        for j in range(m):
            feature1[i*m+j] = brightness_cal[i:i - m + 1 or None, j:j - m + 1 or None]
            feature2[i*m+j] = brightness2_cal[i:i - m + 1 or None, j:j - m + 1 or None]


    output = np.sum(-1/smooth*np.log(np.sqrt(feature1/feature2)+np.sqrt(feature2/feature1)), axis=0)

    return np.exp(output)

def similarity_sar(b1,b2,smooth = 1):
    b1 = np.exp(b1)
    b2 = np.exp(b2)
    return np.exp(np.sum(-1/smooth * np.log(np.sqrt(np.divide(b1,b2))+np.sqrt(np.divide(b2,b1)))))

def dist_sar(b1,b2,smooth = 1):
    b1 = np.exp(b1)
    b2 = np.exp(b2)
    return -np.sum(np.log(np.sqrt(np.divide(b1,b2))+np.sqrt(np.divide(b2,b1))))

def similarity_point_sar_prior(p1, p2, m=3, t=1):
    n = int(math.ceil(m - 1) / 2)

    brightness_cal = np.pad(p1, n, 'edge')
    brightness2_cal = np.pad(p2, n, 'edge')
    feature1 = np.empty((m*m, p1.shape[0], p1.shape[1]))
    feature2 = np.empty((m*m, p1.shape[0], p1.shape[1]))
    for i in range(m):
        for j in range(m):
            feature1[i*m+j] = brightness_cal[i:i - m + 1 or None, j:j - m + 1 or None]
            feature2[i*m+j] = brightness2_cal[i:i - m + 1 or None, j:j - m + 1 or None]

    output = np.sum(-1/t*np.square(feature1-feature2)/(feature1*feature2), axis=0)
    return np.exp(output)

def l2_similarity_point(brightness, brightness2, m=3, smooth=1):
    n = int(math.ceil(m - 1) / 2)

    brightness_cal = np.pad(brightness, n, 'edge')
    brightness2_cal = np.pad(brightness2, n, 'edge')
    feature1 = feature2 = np.array([]).reshape(0, brightness.shape[0], brightness.shape[1])
    for i in range(m):
        for j in range(m):
            feature1 = np.concatenate(
                (feature1, np.expand_dims(brightness_cal[i:i - m + 1 or None, j:j - m + 1 or None], axis=0)), axis=0)
            feature2 = np.concatenate(
                (feature2, np.expand_dims(brightness2_cal[i:i - m + 1 or None, j:j - m + 1 or None], axis=0)), axis=0)

    return np.exp(-1/smooth*np.sqrt(np.sum(np.square(feature1 - feature2),axis=0)))

def point_feature(x, m=7,padding=True):
    n = int(math.ceil(m - 1) / 2)
    if not padding:
        output = np.empty((m**2,x.shape[0]-2*n,x.shape[1]-2*n))
        for i in range(m):
            for j in range(m):
                output[i*m+j] = x[i:i - m + 1 or None, j:j - m + 1 or None]
    else:
        output = np.empty((m**2, x.shape[0], x.shape[1]))
        x = np.pad(x, n, 'symmetric')
        for i in range(m):
            for j in range(m):
                output[i*m+j] = x[i:i - m + 1 or None, j:j - m + 1 or None]


    return output

