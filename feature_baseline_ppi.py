import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from Utils.DataProcessing import *
import scipy
import sys

seed = 2605
l = 9
r = 50
epsilon = float(sys.argv[1])
numbit_whole = 0
numbit_frac = 9
num_file_per_batch = 1000
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
eps = epsilon/r
print(epsilon, eps)
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

float_bin = lambda x: float_to_binary(x, numbit_whole, numbit_frac)
float_to_binary_vec = np.vectorize(float_bin)
bin_float = lambda x: binary_to_float(x, numbit_whole, numbit_frac)
binary_to_float_vec = np.vectorize(bin_float)
save_path = '../Datasets/PPI/feat/'
file_path = '../Datasets/PPI/feat/'
files = [
    'ppi_0_feat_val.npy',
    'ppi_1_feat_val.npy',
    'ppi_0_feat_test.npy',
    'ppi_1_feat_test.npy',
    'ppi_0_feat.npy',
    'ppi_1_feat.npy',
    'ppi_2_feat.npy',
    'ppi_3_feat.npy',
    'ppi_4_feat.npy',
    'ppi_5_feat.npy',
    'ppi_6_feat.npy',
    'ppi_7_feat.npy',
    'ppi_8_feat.npy',
    'ppi_9_feat.npy',
    'ppi_10_feat.npy',
    'ppi_11_feat.npy',
    'ppi_12_feat.npy',
    'ppi_13_feat.npy',
    'ppi_14_feat.npy',
    'ppi_15_feat.npy',
    'ppi_16_feat.npy',
    'ppi_17_feat.npy',
    'ppi_18_feat.npy',
    'ppi_19_feat.npy',
]

# upper_limit = 0
# for i in range(1, numbit_frac+1):
#     upper_limit += (0.5)**i
# print("Upper value for {} number of fractional bit is: {}".format(numbit_frac, upper_limit))

for f in tqdm(files):
    name = f.split('.')[0]
    feat_matrix = np.load(file_path+f)
    print("Size of feature matrix:", feat_matrix.shape)
    max_by_row = np.max(feat_matrix, axis=1)
    min_by_row = np.min(feat_matrix, axis=1)
    range_by_row = np.max(feat_matrix, axis=1) - np.min(feat_matrix, axis=1)
    min_by_row = np.expand_dims(min_by_row, axis=-1)
    range_by_row = np.expand_dims(range_by_row, axis=-1)
    min_by_row = np.tile(min_by_row, (1, r))
    range_by_row = np.tile(range_by_row, (1, r))
    feat_matrix = (feat_matrix - min_by_row)/range_by_row
    del(range_by_row)
    del(min_by_row)
    num_sample = feat_matrix.shape[0]
    perturbed_feat_duchi = duchi_vec(feat_matrix)
    np.save(save_path+'duchi_{}_eps_{}.npy'.format(name, epsilon),perturbed_feat_duchi)
    # piecewise
    perturbed_feat_piecewise = piece_vec(feat_matrix)
    np.save(save_path+'piecewise_{}_eps_{}.npy'.format(name, epsilon),perturbed_feat_piecewise)
    # hybrid
    perturbed_feat_hybrid = hybrid_vec(feat_matrix)
    np.save(save_path+'hybrid_{}_eps_{}.npy'.format(name, epsilon),perturbed_feat_hybrid)
    # threeoutput
    perturbed_feat_threeoutput = threeoutput_vec(feat_matrix)
    np.save(save_path+'threeoutput_{}_eps_{}.npy'.format(name, epsilon),perturbed_feat_threeoutput)
    #PM-SUB
    perturbed_feat_PM = PM_vec(feat_matrix)
    np.save(save_path+'PM_{}_eps_{}.npy'.format(name, epsilon),perturbed_feat_PM)
    del(perturbed_feat_duchi)
    del(perturbed_feat_piecewise)
    del(perturbed_feat_hybrid)
    del(perturbed_feat_threeoutput)
    del(perturbed_feat_PM)

print("==============Process is Done===============")


