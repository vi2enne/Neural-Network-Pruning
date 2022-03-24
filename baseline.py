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

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from network import CompactCNN

# Machine Learning preprocessing and evaluation

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


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


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
if cuda:
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")

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

batch_size = 32 

validation_split=0.1 

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
np.random.seed(42)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=0)
splits = splitter.split(train_set, train_classes)

for train_index, test_index in splits:
    train_set_new = train_set[train_index]
    val_set = train_set[test_index]
    train_classes_new = train_classes[train_index]
    val_classes = train_classes[test_index]

callbacks = None

epochs = 10


# Summary of Training options


print("Batch size:", batch_size, "\nEpochs:", epochs)



def train(model, epoch):

    #for batch_idx, (data, target) in enumerate(zip(train_set, train_classes)):
    n_batches = train_set_new.shape[0]//batch_size
    n_train_sample = train_set_new.shape[0]
    random_idx = np.random.permutation(n_train_sample)

    for batch_idx in range(n_batches):

        inds = random_idx[batch_idx*batch_size : (batch_idx+1)*batch_size]
        data = train_set_new[inds]
        target = train_classes_new[inds]
        target_indx = np.argmax(target, axis=1)

        #if args.cuda:
        data, target = torch.from_numpy(data).cuda(), torch.from_numpy(target_indx).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) 
        loss.mean().backward()
        optimizer.step()

def test(model, test_set, test_classes):

    with torch.no_grad():
        model.eval()
        # compute probabilities for the classes (= get outputs of output layer)
        test_set_tensor = torch.from_numpy(test_set).cuda()
        output = model(test_set_tensor)
        _, predicted = torch.max(output.data, 1)
        test_pred = predicted.data.cpu().numpy()

        # evaluate Accuracy
        test_gt = np.argmax(test_classes, axis=1)
        accuracy = accuracy_score(test_gt, test_pred)
        #print('Accuracy: {}'.format(accuracy) )

        # evaluate Precision
        #print('Precision: {}'.format(precision_score(test_gt, test_pred, average='micro')) )

        # evaluate Recall
        #print('Recall: {}'.format(recall_score(test_gt, test_pred, average='micro')) )

        #print(classification_report(test_gt, test_pred, target_names=metadata.columns))

    return accuracy




print("Test set results")
#print( test(model, test_set, test_classes) )
acc_r = []
for i in range(6):

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    # training
    model = CompactCNN()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.01)

    best_acc = 0.
    for epoch in range(epochs):

        train(model, epoch)
        acc = test(model, val_set, val_classes)
        is_best = acc > best_acc #and epoch>end_pruning_epoch
        best_acc = max(acc, best_acc) #if epoch>end_pruning_epoch else 0
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath='checkpoints/')

    print("Best accuracy: " + str(best_acc) + '\n\n\n')

    # testing
    model = CompactCNN()
    checkpoint = torch.load( os.path.join('checkpoints', 'model_best.pth.tar') )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    acc_r.append(test(model, test_set, test_classes))

print('R_Per = %.3f +- %.3f' % (np.mean(acc_r),np.std(acc_r)))    
print(acc_r)































