# SET PATH OF DOWNLOADED DATA HERE
# (can be relative path if you unzipped the files inside this tutorial's folder)

SPECTROGRAM_PATH = 'ISMIR2018_tut_melspecs_subset'

# included in repository
METADATA_PATH = 'ismir2018_tutorial/NEW_METADATA'# ''

import os
from os.path import join
import shutil

# here, %s will be replace by 'instrumental', 'genres' or 'moods'
LABEL_FILE_PATTERN = join(METADATA_PATH, 'ismir2018_tut_part_1_%s_labels_subset_w_clipid.csv') 
SPECTROGRAM_FILE_PATTERN = join(SPECTROGRAM_PATH, 'ISMIR2018_tut_melspecs_part_1_%s_subset.npz')

# IF YOU USE A GPU, you may set which GPU(s) to use here:
# (this has to be set before the import of Keras and Tensorflow)
os.environ["CUDA_VISIBLE_DEVICES"]="0" #"0,1,2,3"

# General Imports

import argparse
import csv
import datetime
import glob
import math
import sys
import time
import numpy as np
import pandas as pd # Pandas for reading CSV files and easier Data handling in preparation

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from network import CompactCNNPruneScaling, CompactCNNPrune
from ScaleLayer import ScaleLayer
from thop import profile

# Machine Learning preprocessing and evaluation

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--filter-percent', type=float, default=0.7,
                    help='scale sparse rate (default: 0.8)')
parser.add_argument('--model', default='/home/tinghwan/work/genre_pruning/pruned0.94/pruned.pth.tar', type=str, metavar='PATH',
#parser.add_argument('--model', default='/home/tinghwan/work/genre_pruning/checkpoints/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./pruning0.7', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--baseline', action='store_true', default=False,
                    help='test baseline model')
parser.add_argument('--infer', action='store_true', default=True,
                    help='decompress and infer')

end = time.time()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# SET PATH OF DOWNLOADED DATA HERE
# (can be relative path if you unzipped the files inside this tutorial's folder)

SPECTROGRAM_PATH = 'ISMIR2018_tut_melspecs_subset'

# included in repository
METADATA_PATH = 'ismir2018_tutorial/NEW_METADATA'# 'metadata'


# here, %s will be replace by 'instrumental', 'genres' or 'moods'
LABEL_FILE_PATTERN = join(METADATA_PATH, 'ismir2018_tut_part_1_%s_labels_subset_w_clipid.csv') 
SPECTROGRAM_FILE_PATTERN = join(SPECTROGRAM_PATH, 'ISMIR2018_tut_melspecs_part_1_%s_subset.npz')


torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
np.random.seed(42)



if args.model:
    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model)
        if 'cfg' in checkpoint.keys():
            cfg=checkpoint['cfg']
        else:
            cfg = None
        if args.baseline:
            model = CompactCNN(cfg)
        else:
            model = CompactCNNPruneScaling(cfg, 0)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

# se = SizeEstimator(model, input_size=(1,1,80,80))
# tmb, _ = se.estimate_size()
# print("Total in MB {}".format(tmb))
# print("Params in KB {}".format( (se.param_bits/8)/(1024**1) ) ) # bits taken up by parameters
# print("Memory stored for forward and backward in MB {}".format(  (se.forward_backward_bits/8) / (1024**2) ) ) # bits stored for forward and backward
# print(se.input_bits) # bits for input

if args.cuda:
    model.cuda()


# profiling model
cropsize = 80
flops, params = profile(model, input_size=(1, 1, cropsize, cropsize))
print("FLOPS: '{}'".format(flops))
print("Params: '{}'".format(params))

#################### Load Audio Spectrograms ####################


task = 'genres'

# we define the same in a convenience function used later
def load_spectrograms(spectrogram_filename):
    # load spectrograms
    with np.load(spectrogram_filename) as npz:
        spectrograms = npz["features"]
        spec_clip_ids = npz["clip_id"]
    # create dataframe that associates the index order of the spectrograms with the clip_ids
    spectrograms_clip_ids = pd.DataFrame({"spec_id": np.arange(spectrograms.shape[0])}, index = spec_clip_ids)
    spectrograms_clip_ids.index.name = 'clip_id'
    return spectrograms, spectrograms_clip_ids


########################Standardization#################

def standardize(data):
    # vectorize before standardization (cause scaler can't do it in that format)
    N, ydim, xdim = data.shape
    data = data.reshape(N, xdim*ydim)

    # standardize
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)

    # reshape to original shape
    return data.reshape(N, ydim, xdim)

torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
np.random.seed(42)

# load Mel spectrograms
spectrogram_file = SPECTROGRAM_FILE_PATTERN % task
spectrograms, spectrograms_clip_ids = load_spectrograms(spectrogram_file)

# standardize
data = standardize(spectrograms)
data.shape # verify the shape of the loaded & standardize spectrograms

#################### Load the Metadata #####################
# use META_FILE_PATTERN to load the correct metadata file. set correct METADATA_PATH above
csv_file = LABEL_FILE_PATTERN % task
metadata = pd.read_csv(csv_file, index_col=0) #, sep='\t')
metadata.shape

metadata.head()

# how many instrumental tracks
metadata.sum()

# how many vocal tracks
(1-metadata).sum()


# baseline:
metadata.sum().max() / len(metadata)


#################Align Metadata and Spectrograms#################

len(metadata)

# check if we find all metadata clip ids in our spectrogram data
len(set(metadata.index).intersection(set(spectrograms_clip_ids)))


# we may have more spectrograms than metadata
spectrograms.shape

meta_clip_ids = metadata.index
spec_indices = spectrograms_clip_ids.loc[meta_clip_ids]['spec_id']
data = spectrograms[spec_indices,:]

# for training convert from Pandas DataFrame to numpy array
classes = metadata.values

# number of classes is number of columns in metaddata
n_classes = metadata.shape[1]

data = np.expand_dims(data, axis=1)

# we store the new shape of the images in the 'input_shape' variable.
# take all dimensions except the 0th one (which is the number of files)
input_shape = data.shape[1:]  

# use 75% of data for train, 25% for test set
testset_size = 0.25

# Stratified Split retains the class balance in both sets

splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(data, classes)

for train_index, test_index in splits:
    train_set = data[train_index]
    test_set = data[test_index]
    train_classes = classes[train_index]
    test_classes = classes[test_index]

def test(model, test_set, test_classes, pruning_rate):

    # compute probabilities for the classes (= get outputs of output layer)
    model.eval()
    with torch.no_grad():

        for i in range(len(test_classes)):
            print(i)
            test_set_tensor = torch.unsqueeze(torch.from_numpy(test_set[i]).cuda(), 0)

            if args.baseline:
                output = model(test_set_tensor)
            else:
                output, _, _ = model(test_set_tensor, 0, is_training=False)
        # test_set_tensor = torch.from_numpy(test_set).cuda()
        # if args.baseline:
        #     output = model(test_set_tensor)
        # else:
        #     output, _, _ = model(test_set_tensor, 0, is_training=False)

        # _, predicted = torch.max(output.data, 1)
        # test_pred = predicted.data.cpu().numpy()

        # # evaluate Accuracy
        # test_gt = np.argmax(test_classes, axis=1)
        # accuracy = accuracy_score(test_gt, test_pred)
        # print('Accuracy: {}'.format(accuracy) )

        # # evaluate Precision
        # print('Precision: {}'.format(precision_score(test_gt, test_pred, average='micro')) )

        # # evaluate Recall
        # print('Recall: {}'.format(recall_score(test_gt, test_pred, average='micro')) )

        # print(classification_report(test_gt, test_pred, target_names=metadata.columns))

    #return accuracy

# Metrics
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

metrics = ['accuracy', precision, recall]    

if args.infer:
    test(model, test_set, test_classes, 0)
    print('Average time: ', (time.time() - end) / len(test_classes) )
# print("PSNR: " + str(psnr) + " SSIM: " + str(ssim))

# # profiling model
# cropsize = 32
# flops, params = profile(model, input_size=(1, 3, cropsize, cropsize), baseline=args.baseline)
# print("FLOPS: '{}'".format(flops))
# print("Params: '{}'".format(params))

print('Total time: ', time.time() - end)
