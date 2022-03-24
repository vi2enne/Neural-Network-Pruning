# SET PATH OF DOWNLOADED DATA HERE
# (can be relative path if you unzipped the files inside this tutorial's folder)

SPECTROGRAM_PATH = 'ISMIR2018_tut_melspecs_subset'

# included in repository
METADATA_PATH = 'ismir2018_tutorial/NEW_METADATA'# 'metadata'

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
import itertools

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from network import CompactCNNPruneScaling
from thop import profile
from ScaleLayer import ScaleLayer


# Machine Learning preprocessing and evaluation

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--filter-percent', type=float, default=0.8,
                    help='scale sparse rate (default: 0.8)')
parser.add_argument('--model', default='checkpoint.s1e-4p1e-4c1e-4.pruning0.8/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./pruning0.8', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot distributions')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = CompactCNNPruneScaling(filter_percent=args.filter_percent)
if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) ACC: {:f}"
              .format(args.model, checkpoint['epoch'], best_acc))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

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
# Note: this for loop is only executed once if n_splits==1

print(train_set.shape)
print(test_set.shape)

print(model)
total = 0
for m in model.modules():
    if isinstance(m, ScaleLayer):
        total += m.scale.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, ScaleLayer):
        size = m.scale.data.shape[0]
        bn[index:(index+size)] = m.scale.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.filter_percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, ScaleLayer):
        weight_copy = m.scale.data.abs().clone()
        mask = weight_copy.gt(thre.cuda()).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.scale.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

# final fc layer
cfg_mask.append(torch.ones(8))

pruned_ratio = pruned/total
print(pruned_ratio)

print('Pre-processing Successful!')


def test(model, test_set, test_classes, pruning_rate):

    # compute probabilities for the classes (= get outputs of output layer)
    model.eval()
    with torch.no_grad():
        test_set_tensor = torch.from_numpy(test_set).cuda()
        output, _, _ = model(test_set_tensor, pruning_rate, is_training=False)
        _, predicted = torch.max(output.data, 1)
        test_pred = predicted.data.cpu().numpy()

        # evaluate Accuracy
        test_gt = np.argmax(test_classes, axis=1)
        accuracy = accuracy_score(test_gt, test_pred)
        print('Accuracy: {}'.format(accuracy) )

        # # evaluate Precision
        # print('Precision: {}'.format(precision_score(test_gt, test_pred, average='micro')) )

        # # evaluate Recall
        # print('Recall: {}'.format(recall_score(test_gt, test_pred, average='micro')) )

        # print(classification_report(test_gt, test_pred, target_names=metadata.columns))

    return accuracy

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

# Make real prune
print(cfg)
newmodel = CompactCNNPruneScaling(cfg=cfg, filter_percent=args.filter_percent)
if args.cuda:
    newmodel.cuda()
print(newmodel)

layer_id_in_cfg = 0
start_mask = torch.ones(1)
end_mask = cfg_mask[layer_id_in_cfg]
first_BN = True
fc_index = 0

# for [m0, m1] in zip(model.modules(), newmodel.modules()):
old_modules = [module for module in model.modules() if len(list(module.children()))==0]
new_modules = [module for module in newmodel.modules() if len(list(module.children()))==0]

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, ScaleLayer):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.scale.data = m0.scale.data[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]

    elif isinstance(m0, nn.BatchNorm2d):
        if first_BN is not True:
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            first_BN = False

    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()

    elif isinstance(m0, nn.Linear):
        fc_index += 1
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        print('In shape: {:d}.'.format(idx0.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        
        # for feature map not of size 1x1 (7x7 for imagenet 224 cropsize)
        if fc_index==1:
            idx0_new = []
            idx_list = list(idx0)
            idx0_new= [list(range(idx*6, (idx+1)*6) ) for idx in idx_list]
            idx0_new = list(itertools.chain.from_iterable(idx0_new) )
            idx0 = np.array(idx0_new)

            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist()].clone()
            w1 = w1[idx1.tolist(), :].clone()
            m1.weight.data = w1.clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        elif fc_index==2:
            m1.weight.data = m0.weight.data[:, idx0.tolist()].clone()
            m1.bias.data = m0.bias.data.clone()


torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

print(newmodel)
test(model, test_set, test_classes, 0)
test(newmodel, test_set, test_classes, 0)

# profiling model
flops2, params2 = profile(newmodel, input_size=(1, 1, cropsize, cropsize))
print("FLOPS: {}->{}, Compression ratio: {}".format(flops, flops2, float(flops/flops2)))
print("Params: {}->{}, Compression ratio: {}".format(params, params2, float(params/params2)))





































