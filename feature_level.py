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

seed = 2605
l = 9
epsilon = 2.0
gamma = 0.9
numbit_whole = 0
numbit_frac = 9
num_file_per_batch = 1000
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

float_bin = lambda x: float_to_binary(x, numbit_whole, numbit_frac)
float_to_binary_vec = np.vectorize(float_bin)
bin_float = lambda x: binary_to_float(x, numbit_whole, numbit_frac)
binary_to_float_vec = np.vectorize(bin_float)


feat_matrix = np.load('../Datasets/REDDIT/feat.npy')
print("Size of feature matrix:", feat_matrix.shape)
r = feat_matrix.shape[1]


upper_limit = 0
for i in range(1, numbit_frac+1):
    upper_limit += (0.5)**i
print("Upper value for {} number of fractional bit is: {}".format(numbit_frac, upper_limit))

max_by_row = np.max(feat_matrix, axis=1)
min_by_row = np.min(feat_matrix, axis=1)
range_by_row = np.max(feat_matrix, axis=1) - np.min(feat_matrix, axis=1)
min_by_row = np.expand_dims(min_by_row, axis=-1)
range_by_row = np.expand_dims(range_by_row, axis=-1)
min_by_row = np.tile(min_by_row, (1, r))
range_by_row = np.tile(range_by_row, (1, r))
feat_matrix = (feat_matrix - min_by_row)/range_by_row
feat_matrix = np.clip(feat_matrix, a_min = 0, a_max = upper_limit)

distance = 0
mu = epsilon*((l+1)*r + 1)/(4*r*l)
curr_file = 0
batch = 1
for t in tqdm(range(feat_matrix.shape[0])):
    feat = feat_matrix[t]
    theta = np.ones(feat.shape)/r
    binary = float_to_binary_vec(feat)
    temp = "".join(i for i in binary)
    choices = []
    bitstring = []
    indx = range(len(temp))
    for j in indx:
        i = int(np.floor(j/l))
        thet = theta[i]
        eps = epsilon*thet
        p = sigmoid((r*eps*(l-j%l)/l)-mu)
        choice = random.choices([0, 1], weights=[p, 1 - p], k=1)
        choices.append(choice[0])
        bitstring.append(int(temp[j]))
    choices = np.array(choices)
    bitstring = np.array(bitstring)
    perturbed = (choices + bitstring)%2
    temp = []
    a = ''
    for j in indx:
        if (j%l == 0):
            a = ''
            a = a + str(perturbed[j])
        elif (j%l == l-1):
            a = a + str(perturbed[j])
            temp.append(a)
        else:
            a = a + str(perturbed[j])
    
    # print(curr_file, batch)
    if curr_file == 0:
        perturbed_feat = np.expand_dims(binary_to_float_vec(np.array(temp)), axis = 0)
        curr_file += 1
    else:
        perturbed_feat = np.concatenate((perturbed_feat, binary_to_float_vec(np.expand_dims(np.array(temp), axis = 0))), axis=0)
        curr_file += 1
        if (curr_file >= num_file_per_batch): 
            print(perturbed_feat.shape, batch)
            distance += np.mean(np.abs(perturbed_feat - feat))
            np.save('../Datasets/REDDIT/eps2/reddit_perturbed_feat_eps_{}_batch_{}.npy'.format(epsilon, batch), perturbed_feat)
            curr_file = 0
            batch += 1
            del perturbed_feat 
print("Distance from original:", distance/batch)
np.save('../Datasets/REDDIT/eps2/reddit_perturbed_feat_eps_{}_batch_{}.npy'.format(epsilon, batch), perturbed_feat)
print("==============Process is Done===============")



# duchi_mech = lambda x: duchi_mechanism(x, eps)
# duchi_vec = np.vectorize(duchi_mech)
# hybrid_mech = lambda x: hybrid_mechanism(x, eps)
# hybrid_vec = np.vectorize(hybrid_mech)
# piece_mech = lambda x: piecewise_mechanism(x, eps)
# piece_vec = np.vectorize(piece_mech)
# threeoutput_mech = lambda x: three_output_mechanism(x, eps)
# threeoutput_vec = np.vectorize(threeoutput_mech)
# PM_mech = lambda x: PM_SUB(x, eps)
# PM_vec = np.vectorize(PM_mech)

# uplim = 0
# for i in range(1, numbit_frac+1):
#     uplim += (0.5)**i
# print("Upper value for {} number of fractional bit is: {}".format(numbit_frac, uplim))

# min_by_row = np.min(feat_matrix, axis=1)
# range_by_row = np.max(feat_matrix, axis=1) - np.min(feat_matrix, axis=1)
# min_by_row = np.expand_dims(min_by_row, axis=-1)
# range_by_row = np.expand_dims(range_by_row, axis=-1)
# min_by_row = np.tile(min_by_row, (1, r))
# range_by_row = np.tile(range_by_row, (1, r))
# feat_matrix = (feat_matrix - min_by_row)/range_by_row
# feat_matrix = np.clip(feat_matrix, a_min = 0, a_max = uplim)

# feat_matrix = float_to_binary_vec(feat_matrix)
# perturbed_feat_duchi = duchi_vec(feat_matrix)
# # perturbed_feat_duchi = binary_to_float_vec(perturbed_feat_duchi)
# print(np.min(perturbed_feat_duchi), np.max(perturbed_feat_duchi), np.mean(np.abs(perturbed_feat_duchi - feat_matrix)))
# np.save('../Datasets/MIR_final/duchi_eps_' + str(epsilon) + '.npy', perturbed_feat_duchi)     
# print("Process is Done for Duchi")
# perturbed_feat_piecewise = piece_vec(feat_matrix)
# # perturbed_feat_piecewise = binary_to_float_vec(perturbed_feat_piecewise)
# print(np.min(perturbed_feat_piecewise), np.max(perturbed_feat_piecewise), np.mean(np.abs(perturbed_feat_piecewise - feat_matrix)))
# np.save('../Datasets/MIR_final/piecewise_eps_' + str(epsilon) + '.npy', perturbed_feat_piecewise)     
# print("Process is Done for Piecewise")
# perturbed_feat_hybrid = hybrid_vec(feat_matrix)
# # perturbed_feat_hybrid = binary_to_float_vec(perturbed_feat_hybrid)
# print(np.min(perturbed_feat_hybrid), np.max(perturbed_feat_hybrid), np.mean(np.abs(perturbed_feat_hybrid - feat_matrix)))
# np.save('../Datasets/MIR_final/hybrid_eps_' + str(epsilon) + '.npy', perturbed_feat_hybrid)     
# print("Process is Done for Hybrid")
# perturbed_feat_threeoutput = threeoutput_vec(feat_matrix)
# # perturbed_feat_threeoutput = binary_to_float_vec(perturbed_feat_threeoutput)
# print(np.min(perturbed_feat_threeoutput), np.max(perturbed_feat_threeoutput), np.mean(np.abs(perturbed_feat_threeoutput - feat_matrix)))
# np.save('../Datasets/MIR_final/threeoutput_eps_' + str(epsilon) + '.npy', perturbed_feat_threeoutput)     
# print("Process is Done for Three output")
# perturbed_feat_PM = PM_vec(feat_matrix)
# print(np.min(perturbed_feat_PM), np.max(perturbed_feat_PM), np.mean(np.abs(perturbed_feat_PM - feat_matrix)))
# # perturbed_feat_PM = binary_to_float_vec(perturbed_feat_PM)
# np.save('../Datasets/MIR_final/PM_eps_' + str(epsilon) + '.npy', perturbed_feat_PM)     
# print("Process is Done for PM mechanism")

# print(perturbed_feat.shape, np.max(perturbed_feat), np.min(perturbed_feat))
# np.save('../Datasets/MIR_FLICKR/feat/perturbed_feat_eps_' + str(epsilon) + '_gamma_' + str(gamma) + '_numbit_' + str(l) + '.npy', perturbed_feat)     

# print(float_to_binary_vec(1.0 - 1e-3))

