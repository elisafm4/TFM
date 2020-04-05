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

import trainer.model as model

backend.set_floatx('float32')
backend.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_and_evaluate(args):
    nummodal = int(len(args['modalities']))
    padsize = np.max(np.array(args['patchsize']) + 1) / 2
    args['batchsize'] = args['batchsize'] * len(args['gpu_ids'])
    f = 0
    for i in range(args['numatlas']):
        maskname = model.check_nifti_filepath(args['atlasdir'], ('atlas%d' % (i + 1)) + '_' + 'lesion')
        f += min(args['max_patches'], nib.load(maskname).get_data().sum())

    print('Total number of lesion patches = ' + str(int(f)))
    r = 1 # ratio between positive and negative patches
    mask_patches = np.zeros((int((r+1)*f),) + args['patchsize'] + (1,), dtype=np.float32)
    image_patches = np.zeros((int((r+1)*f),) + args['patchsize'] + (nummodal,), dtype=np.float32)
    
    time_id = time.strftime('%d-%m-%Y_%H-%M-%S')
    print('Unique ID is %s ' % time_id)
    con = '+'.join([str(mod).upper() for mod in args['modalities']])
    psize = 'x'.join([str(side) for side in args['patchsize']])
    outname = 'FLEXCONN_Model_' + psize + '_Orient%d%d%d_' + con + '_' + time_id + '.h5'
    
    codes = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    
    for orient in range(1 if args['axial_only'] else 3):
        transpose_code = codes[orient]
        orient_outname = os.path.join(args['outdir'], outname % transpose_code)
        if args['loss'] == 'bce':
            tempoutname = orient_outname.replace('.h5', '_epoch-{epoch:03d}_acc-{val_acc:.4f}.h5')
        else:
            tempoutname = orient_outname.replace('.h5', '_epoch-{epoch:03d}_val_loss-{val_loss:.4f}.h5')

        print('Model for orientation %s will be written at %s' % (str(transpose_code), orient_outname))

        # Re-initialize total matrix size because some of them could be discarded during multi-gpu process
        mask_patches = np.zeros((int((r+1) * f),) + args['patchsize'] + (1,), dtype=np.float32)
        image_patches = np.zeros((int((r+1) * f),) + args['patchsize'] + (nummodal,), dtype=np.float32)

        patch_count = 0
        for i in range(args['numatlas']):
            #segpath = check_nifti_filepath(args['atlasdir'], ('atlas%02d' % (i + 1)) + '_' + 'lesion')
            segpath = model.check_nifti_filepath(args['atlasdir'], ('atlas%d' % (i + 1)) + '_' + 'lesion')
            mask = np.transpose(pad_image(nib.load(segpath).get_data(), padsize),
                                axes=transpose_code).astype(np.float32)
            vol4d = np.zeros(mask.shape + (nummodal,), dtype=np.float32)
            
            for j in range(nummodal):
                filepath = model.check_nifti_filepath(args['atlasdir'],
                                                ('atlas%d' % (i + 1)) + '_' + args['modalities'][j].upper())
                #                                ('atlas%02d' % (i + 1)) + '_' + args['modalities'][j].lower())
                print('Reading %s' % filepath)
                vol4d[:, :, :, j] = np.transpose(pad_image(normalize_image(nib.load(filepath).get_data(),args['modalities'][j]), padsize),
                                                 axes=transpose_code).astype(np.float32)
            print('Atlas %d size = %d x %d x %d x %d ' % ((i+1,) + vol4d.shape))

            image_patches_a, mask_patches_a = get_patches(vol4d, mask, args)
            num_patches = image_patches_a.shape[0]
            print('Atlas %d : indices [%d,%d)' % (i+1, patch_count, patch_count + num_patches))
            image_patches[patch_count:patch_count + num_patches, :, :, :] = image_patches_a
            mask_patches[patch_count:patch_count + num_patches, :, :, :] = mask_patches_a
            patch_count += num_patches
            print('-' * 100)

        image_patches = image_patches[0:patch_count, :, :, :]
        mask_patches = mask_patches[0:patch_count, :, :, :]

        """
        If the number of patches within training & validation sets are not multiple of number of GPU,
        there could be some arbitrary CUDNN error. Although ideally this should be taken care within
        multi_gpu_model function, it is not; discarding last couple of samples is easiest.
        This will not work if the number of samples is low, e.g. marmosets or CT. Use single gpu in those cases.
        """
        numgpu = args['numgpu']
        if numgpu>1:
            L = image_patches.shape[0]
            L = np.floor(np.floor(L * 0.2) / numgpu) * numgpu * 5
            L = np.asarray(L, dtype=int)
            image_patches = image_patches[0:L, :, :, :]
            mask_patches = mask_patches[0:L, :, :, :]

        print('Total number of patches collected = ' + str(patch_count))
        print('Sizes of the input matrices are ' + str(image_patches.shape) + ' and ' + str(mask_patches.shape))

        flexconn_model = model.get_model_3d(args['base_filters'], nummodal, args['numgpu'], args['loss'])

    os.makedirs(args.outdir)
    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = args.outdir
    
    # Model checkpoint callback.
    if args['loss'] == 'bce':
            callbacks = [ModelCheckpoint(tempoutname, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     period=args['period'], mode='max')] if args['period'] > 0 else None
            dlr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2,
                                mode='max', verbose=1, cooldown=2, min_lr=1e-8)
            earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0002, patience=5,
                                  verbose=1, mode='max')
    else:

            callbacks = [ModelCheckpoint(tempoutname, monitor='val_loss', verbose=1, save_best_only=True,
                                         period=args['period'], mode='min')] if args['period'] > 0 else None
            dlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                    mode='min', verbose=1, cooldown=2, min_lr=1e-8)
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=10,
                                      verbose=1, mode='min')
        callbacks.append(dlr)
        callbacks.append(earlystop)
    
    # Continuous eval callback.

    if args['initmodel'] != 'None' and os.path.exists(args['initmodel']):
            dict = {"tf": tf,
                    }
            oldmodel = load_model(args['initmodel'],custom_objects=dict)
            model.set_weights(oldmodel.get_weights())
            print("Initializing from existing model %s" % (args['initmodel']))

        flexconn_model.fit(image_patches, mask_patches, batch_size=args['batchsize'], epochs=args['epoch'], verbose=1,
                  validation_split=0.2, callbacks=callbacks, shuffle=True)
        
    flexconn_model.fit(image_patches, mask_patches, batch_size=args['batchsize'], epochs=args['epoch'], verbose=1,
                  validation_split=0.2, callbacks=callbacks, shuffle=True)
        

    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if args.outdir.startswith('gs://'):
        flexconn_model.save(outname)
        copy_file_to_gcs(args.outdir, outname)
    else:
        flexconn_model.save((filepath=orient_outname)

model.to_savedmodel(flexconn_model, os.path.join(args.outdir, 'export'))
    

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as fp:
            fp.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FLEXCONN Segmentation Training')
    
    parser.add_argument('--atlasdir', required=True,
                        help='Directory containing atlas images. Images should be in NIFTI (.nii or .nii.gz) and be '
                             'N4 bias corrected. Atlas images should be in the same orientation as the subject'
                             '(axial [RAI]). Atlases should be named atlas{NUMBER}_{MODALITY}.nii.gz. '
                             '{MODALITY} should match the strings entered in the "--modalities" argument. Example '
                             'atlas image names are atlas1_T1.nii.gz, atlas1_T2.nii.gz, atlas1_FL.nii.gz, atlas1_lesion.nii.gz, with '
                             'modalities as --modalities t1 t2 fl.')
    parser.add_argument('--natlas', required=True, type=int,
                        help='Number of atlases to be used. The program will pick the first N atlases from the '
                             'atlas directory. The atlas directory must contain at least this many atlas sets.')
    parser.add_argument('--psize', nargs='+', type=int, default=[100, 100],
                        help='Patch size, e.g. 25 25 25 (3D) or 100 100 (2D). Patch sizes are separated by space. '
                             'Note that bigger patches (such as 128x128) are possible in 2D models while it is '
                             'computationally expensive to use more than 25x25x25 patches. Default is [100, 100] which '
                             'will use 2D patches.')
    parser.add_argument('--modalities', required=True, nargs='+',
                        help='A space separated string of input image modalities. This is used to determine the order '
                             'of images used in training. It also defines how many modalities will be used by training '
                             '(if the atlas directory contains more). Accepted modalities are T1/T1C/T2/PD/FL/FLC. T1C '
                             'and FLC corresponds to postcontrast T1 and FL.')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='Mini-batch size to use per iteration. Usually 32-256 works well. '
                             'Optional argumet, if omitted, 64 is used as default. Both too large or too small '
                             'batches can incur in bad optimization.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs to train. Usually 10-20 works well. Optional argumet, '
                             'if omitted, 20 is used as default. Too large epochs can incur in overfitting.')
    parser.add_argument('--outdir', required=True,
                        help='Output directory where the trained models are written.')
    parser.add_argument('--save', type=int, default=1,
                        help='When training with large number of epochs, the interim models can be saved after every N '
                             'epochs. This option (e.g., --save 4) enables saving the model every 4th epoch.')
    parser.add_argument('--axialonly', action='store_true', default=False,
                        help='Only train a 2D modal in the input (normally axial) orientation. This is common for CT '
                             'where images can be highly anisotropic. This also works for very thick-sliced MR images. '
                             'Without this option, the training images are reoriented in all 3 orientations and '
                             'individual trainings are done for each orientation separately.')
    parser.add_argument('--numgpu', type=int, default=1,
                        help='Number of GPUs to use for training. The program will use the first N visible GPUs. To '
                             'select specific gpus, use "--gpuids" ')
    parser.add_argument('--gpuids', type=int, nargs='+',
                        help='Specifc GPUs to use for training, separated by space. E.g., --gpuids 3 4 5 ')
    parser.add_argument('--basefilters', type=int, default=8,
                        help='Sets the base number of filters for the models. 16 is appropriate for 12GB GPUs, where '
                             '8 may be more appropriate for 4GB cards. This value scales all filter banks (increasing '
                             'by 2 for a change from 8->16)')
    parser.add_argument('--maxpatches', type=int, default=150000,
                        help='Maximum number of patches to choose from each patient. 150000 is the default. This '
                             'is appropriate for 2D patches. 50000 may be more appropriate for 3D patches. This '
                             'value is abitrary and should be scaled to fit the available RAM.')
    parser.add_argument('--loss', type=str, default='mse',
                        help="Loss function to be used during training. Available options are mae (mean absolute error), "
                             "mse(mean squared error), and bce (binary cross-entropy). If mse/mae are chosen, the "
                             "binary lesion masks are first blurred by a Gaussian to compute a membership.")
    parser.add_argument('--initmodel', type=str, dest='INITMODEL', required=False,
                        help='Existing trained model. If provided, the weights will be '
                             'used to initiate the training.')
                             
    results = parser.parse_args()


    if results.gpuids is not None:
        gpu_ids = results.gpuids
    else:
        gpu_ids = range(results.numgpu)
    
        numgpu = len(gpu_ids)
    results.batchsize = (results.batchsize // numgpu) * numgpu
    # Patch size must be odd
    for i in range(len(results.psize)):
        results.psize[i] = (results.psize[i]//2)*2 + 1

    if results.INITMODEL is None:
        results.INITMODEL = 'None'

    loss=['bce','mae', 'mse']
    opt = {'numatlas': results.natlas,
           'outdir': os.path.abspath(os.path.expanduser(results.outdir)),
           'modalities': [item.upper() for item in results.modalities],
           'patchsize': tuple(results.psize),
           'atlasdir': os.path.abspath(os.path.expanduser(results.atlasdir)),
           'batchsize': results.batchsize,
           'epoch': results.epoch,
           'period': results.save,
           'axial_only': results.axialonly,
           'base_filters': results.basefilters,
           'max_patches': results.maxpatches,
           'gpu_ids': gpu_ids,
           'numgpu': numgpu,
           'loss': str(results.loss).lower(),
           'initmodel': results.INITMODEL,
           }
    if opt['loss'] not in loss:
        print('Available loss options are MAE (mean absolute error), MSE (mean squared error), and BCE (binary cross-entropy). Using BCE.')
        opt['loss'] = 'bce'
    
    if not os.path.isdir(opt['outdir']):
        print('Output directory does not exist. I will create it.')
        os.makedirs(opt['outdir'])
    

    
    train_and_evaluate(opt)