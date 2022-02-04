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
gamma = 0.9
numbit_whole = 0
numbit_frac = 9
num_file_per_batch = 1000
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def string_to_int(a):
    bit_str = "".join(x for x in a)
    return np.array(list(bit_str)).astype(int)

def join_string(a, num_bit=l, num_feat=r):
    res = []
    for i in range(num_feat):
        res.append("".join(str(x) for x in a[i*l:(i+1)*l]))
    return np.array(res)

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

upper_limit = 0
for i in range(1, numbit_frac+1):
    upper_limit += (0.5)**i
print("Upper value for {} number of fractional bit is: {}".format(numbit_frac, upper_limit))

for f in tqdm(files):
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
    feat_matrix = np.clip(feat_matrix, a_min = 0, a_max = upper_limit)
    del(range_by_row)
    del(min_by_row)
    distance = 0
    mu = epsilon*((l+1)*r + 1)/(4*r*l)
    num_sample = feat_matrix.shape[0]
    feat = float_to_binary_vec(feat_matrix)
    feat = np.apply_along_axis(string_to_int, axis=1, arr=feat)
    overall_score = epsilon*(np.ones(feat.shape)/r)
    print(overall_score.shape)
    index_matrix = np.array(range(l))
    index_matrix = np.tile(index_matrix, (num_sample, r))
    # overall_score = np.repeat(overall_score, l, axis=1)
    # print(overall_score.shape, index_matrix.shape)
    p = sigmoid(((l - index_matrix)*r/l)*overall_score - mu)
    del(index_matrix)
    del(overall_score)
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    del(p)
    del(p_temp)
    perturb_feat = (perturb + feat)%2
    del(perturb)
    del(feat)
    perturb_feat = np.apply_along_axis(join_string, axis=1, arr=perturb_feat)
    perturb_feat = binary_to_float_vec(perturb_feat)
    np.save(save_path+'perturb' + f.split('.')[0] + 'eps_{}.npy'.format(epsilon), perturb_feat)
    print(f, np.mean(np.abs(perturb_feat - feat_matrix)))
print("==============Process is Done===============")
