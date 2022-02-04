import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
# import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from Utils.DataProcessing import *
import scipy
import sys

seed = 2605
l = 9
# r = 602
epsilon = float(sys.argv[1])
# eps = epsilon/r
numbit_whole = 0
numbit_frac = 9
batch_size = 5000
np.random.seed(seed)
float_bin = lambda x: float_to_binary(x, numbit_whole, numbit_frac)
float_to_binary_vec = np.vectorize(float_bin)
bin_float = lambda x: binary_to_float(x, numbit_whole, numbit_frac)
binary_to_float_vec = np.vectorize(bin_float)

duchi_save_path = '../Datasets/REDDIT/duchi_eps_{}'.format(epsilon)
hybrid_save_path = '../Datasets/REDDIT/hybrid_eps_{}'.format(epsilon)
piecewise_save_path = '../Datasets/REDDIT/piecewise_eps_{}'.format(epsilon)
threeoutput_save_path = '../Datasets/REDDIT/threeoutput_eps_{}'.format(epsilon)
PM_save_path = '../Datasets/REDDIT/PM_eps_{}'.format(epsilon)
if (os.path.isdir(duchi_save_path) == False):
    os.makedirs(duchi_save_path) 
if (os.path.isdir(hybrid_save_path) == False):
    os.makedirs(hybrid_save_path) 
if (os.path.isdir(piecewise_save_path) == False):
    os.makedirs(piecewise_save_path) 
if (os.path.isdir(threeoutput_save_path) == False):
    os.makedirs(threeoutput_save_path) 
if (os.path.isdir(PM_save_path) == False):
    os.makedirs(PM_save_path) 

feat_matrix = np.load('../Datasets/REDDIT/feat.npy')
r = feat_matrix.shape[1]
eps = epsilon/r
print(epsilon, eps)
print("Size of feature matrix:", feat_matrix.shape)
# exit()
duchi_mech = lambda x: duchi_mechanism(x, eps)
duchi_vec = np.vectorize(duchi_mech)
hybrid_mech = lambda x: hybrid_mechanism(x, eps)
hybrid_vec = np.vectorize(hybrid_mech)
piece_mech = lambda x: piecewise_mechanism(x, eps)
piece_vec = np.vectorize(piece_mech)
threeoutput_mech = lambda x: three_output_mechanism(x, eps)
threeoutput_vec = np.vectorize(threeoutput_mech)
PM_mech = lambda x: PM_SUB(x, eps)
PM_vec = np.vectorize(PM_mech)

num_batch = int(feat_matrix.shape[0]/batch_size) + 1

min_by_row = np.min(feat_matrix, axis=1)
range_by_row = np.max(feat_matrix, axis=1) - np.min(feat_matrix, axis=1)
min_by_row = np.expand_dims(min_by_row, axis=-1)
range_by_row = np.expand_dims(range_by_row, axis=-1)
min_by_row = np.tile(min_by_row, (1, r))
range_by_row = np.tile(range_by_row, (1, r))
feat_matrix = (feat_matrix - min_by_row)/range_by_row
del(range_by_row)
del(min_by_row)

for batch in tqdm(range(num_batch)):
    feat = feat_matrix[batch*batch_size:(batch+1)*batch_size,:]
    num_sample = feat.shape[0]
    # duchi
    perturbed_feat_duchi = duchi_vec(feat)
    np.save(duchi_save_path+'/batch_{}.npy'.format(batch),perturbed_feat_duchi)
    # piecewise
    perturbed_feat_piecewise = piece_vec(feat)
    np.save(piecewise_save_path+'/batch_{}.npy'.format(batch),perturbed_feat_piecewise)
    # hybrid
    perturbed_feat_hybrid = hybrid_vec(feat)
    np.save(hybrid_save_path+'/batch_{}.npy'.format(batch),perturbed_feat_hybrid)
    # threeoutput
    perturbed_feat_threeoutput = threeoutput_vec(feat)
    np.save(threeoutput_save_path+'/batch_{}.npy'.format(batch),perturbed_feat_threeoutput)
    #PM-SUB
    perturbed_feat_PM = PM_vec(feat)
    np.save(PM_save_path+'/batch_{}.npy'.format(batch),perturbed_feat_PM)
    del(perturbed_feat_duchi)
    del(perturbed_feat_piecewise)
    del(perturbed_feat_hybrid)
    del(perturbed_feat_threeoutput)
    del(perturbed_feat_PM)

print("==============Process is Done===============")



