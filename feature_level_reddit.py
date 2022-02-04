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
r = 602
epsilon = float(sys.argv[1])
gamma = 0.9
numbit_whole = 0
numbit_frac = 9
batch_size = 5000
np.random.seed(seed)


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
save_path = '../Datasets/REDDIT/perturb_eps_{}'.format(epsilon)
if (os.path.isdir(save_path) == False):
    os.makedirs(save_path) 
feat_matrix = np.load('../Datasets/REDDIT/feat.npy')
num_batch = int(feat_matrix.shape[0]/batch_size) + 1

uplim = 0
for i in range(1, numbit_frac+1):
    uplim += (0.5)**i
print("Upper value for {} number of fractional bit is: {}".format(numbit_frac, uplim))

min_by_row = np.min(feat_matrix, axis=1)
range_by_row = np.max(feat_matrix, axis=1) - np.min(feat_matrix, axis=1)
min_by_row = np.expand_dims(min_by_row, axis=-1)
range_by_row = np.expand_dims(range_by_row, axis=-1)
min_by_row = np.tile(min_by_row, (1, r))
range_by_row = np.tile(range_by_row, (1, r))
feat_matrix = (feat_matrix - min_by_row)/range_by_row
feat_matrix = np.clip(feat_matrix, a_min = 0, a_max = uplim)
del(range_by_row)
del(min_by_row)

mu = epsilon*((l+1)*r + 1)/(4*r*l)
distance = 0
for batch in tqdm(range(num_batch)):
    feat = feat_matrix[batch*batch_size:(batch+1)*batch_size,:]
    num_sample = feat.shape[0]
    feat = float_to_binary_vec(feat)
    feat = np.apply_along_axis(string_to_int, axis=1, arr=feat)
    overall_score = epsilon*(np.ones(feat.shape)/r)
    index_matrix = np.array(range(l))
    index_matrix = np.tile(index_matrix, (num_sample, r))
    p = sigmoid(((l - index_matrix)*r)*overall_score/l - mu)
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
    np.save(save_path+'/batch_{}.npy'.format(batch),perturb_feat)
    distance += np.mean(np.abs(perturb_feat - feat_matrix[batch*batch_size:(batch+1)*batch_size,:]))


print("Distance from original:", distance/num_batch)
print("==============Process is Done===============")




