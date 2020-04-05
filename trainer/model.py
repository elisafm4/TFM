from __future__ import print_function, division

import argparse
import os
import re
import random
import time
import statsmodels.api as sm
import nibabel as nib
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend,losses
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import (AveragePooling2D, AveragePooling3D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, concatenate, Lambda)
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import load_model
import sys
import copy
from tqdm import tqdm
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from scipy.signal import argrelextrema

backend.set_floatx('float32')
backend.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def check_nifti_filepath(directory, file_prefix):
    filepath = os.path.join(directory, file_prefix + '.nii.gz')
    filepath = filepath if os.path.exists(filepath) else os.path.join(directory, file_prefix + '.nii')
    if not os.path.exists(filepath):
        raise ValueError('File %s does not exists' % filepath)
    return filepath


def get_inception_3d(inlayer, base_filters):
    conv_inception1a = Conv3D(base_filters * 4, (1, 1, 1), activation='relu', padding='same')(inlayer)
    
    conv_inception2a = Conv3D(base_filters * 6, (1, 1, 1), activation='relu', padding='same')(inlayer)
    conv_inception4a = Conv3D(base_filters * 8, (3, 3, 3), activation='relu', padding='same')(conv_inception2a)
    
    conv_inception3a = Conv3D(base_filters, (1, 1, 1), activation='relu', padding='same')(inlayer)
    conv_inception5a = Conv3D(base_filters * 2, (5, 5, 5), activation='relu', padding='same')(conv_inception3a)
    
    pool_inception1a = AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inlayer)
    conv_inception6a = Conv3D(base_filters * 2, (1, 1, 1), activation='relu', padding='same')(pool_inception1a)
    
    pool_inception2a = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inlayer)
    conv_inception7a = Conv3D(base_filters * 2, (1, 1, 1), activation='relu', padding='same')(pool_inception2a)
    print('concat')
    outlayer = concatenate([conv_inception1a, conv_inception4a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception5a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception6a], axis=-1)
    outlayer = concatenate([outlayer, conv_inception7a], axis=-1)
    return outlayer


def get_model_3d(base_filters, numchannel, gpus,loss):
    inputs = Input((None, None, None, int(numchannel)))
    conv1 = Conv3D(base_filters * 8, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(inputs)
    conv2 = Conv3D(base_filters * 8, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv1)
    print('first inception')
    
    inception1 = get_inception_3d(conv2, base_filters)
    inception2 = get_inception_3d(inception1, base_filters)
    inception3 = get_inception_3d(inception2, base_filters)
    
    convconcat1 = Conv3D(base_filters * 4, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(inception3)
    convconcat2 = Conv3D(base_filters * 4, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(convconcat1)
    print('check loss')
    if loss == 'bce':
        conv_last = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', strides=(1, 1, 1))(convconcat2)
    else:
        conv_last = Conv3D(1, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(convconcat2)
    model = Model(inputs=inputs, outputs=conv_last)
    print(gpus)
   
    print('compile model')
    if loss == 'bce':
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    elif loss == 'mae':
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_absolute_error')
    else:
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model




def pad_image(vol, padsize):
    dim = vol.shape
    padsize = np.asarray(padsize, dtype=int)
    dim2 = dim + 2 * padsize
    temp = np.zeros(dim2, dtype=np.float32)
    temp[padsize:dim[0] + padsize, padsize:dim[1] + padsize, padsize:dim[2] + padsize] = vol
    return temp


def normalize_image(vol, contrast):
    # All MR images must be non-negative. Sometimes cubic interpolation may introduce negative numbers.
    # This will also affect if the image is CT, which not considered here. Non-negativity is required
    # while getting patches, where nonzero voxels are considered to collect patches.
    vol[vol<0] = 0
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 1.00
    print("%d peaks found." % (len(peaks)))


    if contrast.lower() == "t1" or contrast.lower() == "t1c":
        print("Double checking peaks with a GMM.")
        gmm = GaussianMixture(n_components=3, covariance_type='spherical', tol=0.001,
                reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', precisions_init=None,
                weights_init=(0.33, 0.33, 0.34), means_init=np.reshape((0.2 * q, 0.5 * q, 0.95 * q), (3, 1)),
                warm_start=False, verbose=1, verbose_interval=1)
        gmm.fit(temp.reshape(-1, 1))
        m = gmm.means_[2]
        peak = peaks[-1]
        if m / peak < 0.75 or m / peak > 1.25:
            print("WARNING: WM peak could be incorrect (%.4f vs %.4f). Please check." % (m, peak))
            peaks = m
        peak = peaks[-1]
        print("Peak found at %.4f for %s" % (peak, contrast))
    elif contrast.lower() in ['t2', 'pd', 'fl', 'flc']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        print("Peak found at %.4f for %s" % (peak, contrast))
    else:
        print("Contrast must be either T1,T1C,T2,PD,FL, or FLC. You entered %s. Returning 1." % contrast)
    return vol/peak


def get_patches(vol4d, mask, opt):

    patchsize = opt['patchsize']
    nummodal = len(opt['modalities'])
    maxpatch = opt['max_patches']
    patchsize = np.asarray(patchsize, dtype=int)
    dsize = np.floor(patchsize / 2).astype(dtype=int)
    mask = np.asarray(mask, dtype=np.float32)
    rng = random.SystemRandom()

    if opt['loss'] == 'mse' or opt['loss'] == 'mae':
        if len(patchsize) == 3:
            blurmask = ndimage.filters.gaussian_filter(mask, sigma=(1, 1, 1))
        else:
            blurmask = np.zeros(mask.shape, dtype=np.float32)
            for t in range(0, mask.shape[2]):
                if np.ndarray.sum(mask[:, :, t]) > 0:
                    blurmask[:, :, t] = ndimage.filters.gaussian_filter(mask[:, :, t], sigma=(1, 1))

        blurmask = np.ndarray.astype(blurmask, dtype=np.float32)
        blurmask[blurmask < 0.0001] = 0
        blurmask = blurmask * 100  # Just to have reasonable looking error values during training, no other reason.
    else:
        blurmask = mask
    
    indx = np.nonzero(mask) # indx for positive patches
    indx = np.asarray(indx, dtype=int)


    num_patches = np.minimum(maxpatch, len(indx[0]))
    print('Number of patches used  = %d (out of %d, maximum %d)' % (num_patches, len(indx[0]), maxpatch))
    randindx = random.sample(range(0, len(indx[0])), num_patches)
    newindx = np.ndarray((3, num_patches))
    for i in range(0, num_patches):
        for j in range(0, 3):
            newindx[j, i] = indx[j, randindx[i]]
    newindx = np.asarray(newindx, dtype=int)

    # Add some negative samples as well
    r = 1 # Sampling ratio
    temp = copy.deepcopy(vol4d[:, :, :, 0])
    temp[temp > 0] = 1
    temp[temp <= 0] = 0
    temp = np.multiply(temp, 1 - mask)
    indx0 = np.nonzero(temp)
    indx0 = np.asarray(indx0, dtype=int)
    L = len(indx0[0])

    # Sample equal number of negative patches
    randindx0 = rng.sample(range(0, L), r * num_patches)
    newindx0 = np.ndarray((3, r * num_patches))
    for i in range(0, r * num_patches):
        for j in range(0, 3):
            newindx0[j, i] = indx0[j, randindx0[i]]
    newindx0 = np.asarray(newindx0, dtype=int)

    newindx = np.concatenate([newindx, newindx0], axis=1)


    if len(patchsize) == 2:
        matsize1 = ((r+1) * num_patches, patchsize[0], patchsize[1], 1)
        matsize2 = ((r+1) * num_patches, patchsize[0], patchsize[1], nummodal)
        
        image_patches = np.ndarray(matsize2, dtype=np.float32)
        mask_patches = np.ndarray(matsize1, dtype=np.float32)
        
        for i in range(0, (r+1)*num_patches):
            idx1 = newindx[0, i]
            idx2 = newindx[1, i]
            idx3 = newindx[2, i]
            
            for m in range(0, nummodal):
                image_patches[i, :, :, m] = vol4d[idx1 - dsize[0]:idx1 + dsize[0] + 1,
                                                  idx2 - dsize[1]:idx2 + dsize[1] + 1, idx3, m]
            mask_patches[i, :, :, 0] = blurmask[idx1 - dsize[0]:idx1 + dsize[0] + 1,
                                                idx2 - dsize[1]:idx2 + dsize[1] + 1, idx3]
    else:
        matsize1 = ((r+1)*num_patches, patchsize[0], patchsize[1], patchsize[2], 1)
        matsize2 = ((r+1)*num_patches, patchsize[0], patchsize[1], patchsize[2], nummodal)
        
        image_patches = np.ndarray(matsize2, dtype=np.float32)
        mask_patches = np.ndarray(matsize1, dtype=np.float32)
        
        for i in range(0, (r+1)*num_patches):
            idx1 = newindx[0, i]
            idx2 = newindx[1, i]
            idx3 = newindx[2, i]
            
            for m in range(0, nummodal):
                image_patches[i, :, :, :, m] = vol4d[idx1 - dsize[0]:idx1 + dsize[0] + 1,
                                                     idx2 - dsize[1]:idx2 + dsize[1] + 1,
                                                     idx3 - dsize[2]:idx3 + dsize[2] + 1, m]
            mask_patches[i, :, :, :, 0] = blurmask[idx1 - dsize[0]:idx1 + dsize[0] + 1,
                                                   idx2 - dsize[1]:idx2 + dsize[1] + 1,
                                                   idx3 - dsize[2]:idx3 + dsize[2] + 1]
    return image_patches, mask_patches

def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(
        inputs={'input': model.inputs[0]}, outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            })
        builder.save()