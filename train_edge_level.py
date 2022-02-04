import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import networkx as nx
import random
import sys
import scipy
import matplotlib.pyplot as plt
import zipfile
from Utils.DataProcessing import *
from sklearn.metrics import f1_score
import tensorflow.keras.backend as K

# Parameters
dropout = 0.1          # Dropout rate for the features
l2_reg = 5e-4 / 2      # L2 regularization rate
epochs = 500
patience = 50
seed = 23
num_run = 5
label = 'people'
data_path = '../Datasets/MIR_FLICKR/feat/'
channels = 256
learning_rate = 0.001
num_nodes = np.load('../Datasets/MIR_FLICKR/feat/feat.npy').shape[0]
num_feats = np.load('../Datasets/MIR_FLICKR/feat/feat.npy').shape[1]

print("Number of node:", num_nodes)
print("Number of features:", num_feats)

# seeding
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# read arguments
file_path = ['resnet50_plain_feat.npy',
            'perturbed_feat_eps_1_gamma_0.1.npy',          
            'perturbed_feat_eps_1_gamma_0.2.npy',
            'perturbed_feat_eps_1_gamma_0.3.npy',
            'perturbed_feat_eps_1_gamma_0.4.npy',
            'perturbed_feat_eps_1_gamma_0.5.npy',
            'perturbed_feat_eps_1_gamma_0.6.npy',
            'perturbed_feat_eps_1_gamma_0.7.npy',
            'perturbed_feat_eps_1_gamma_0.8.npy',
            'perturbed_feat_eps_1_gamma_0.9.npy',
            'perturbed_feat_eps_1_gamma_1.0.npy',
            'perturbed_feat_eps_2_gamma_0.5.npy',
            'perturbed_feat_eps_3_gamma_0.5.npy',
            'perturbed_feat_eps_4_gamma_0.5.npy',
            'perturbed_feat_eps_5_gamma_0.5.npy',
            'perturbed_feat_eps_6_gamma_0.5.npy',
            'perturbed_feat_eps_7_gamma_0.5.npy',
            'perturbed_feat_eps_8_gamma_0.5.npy',
            'perturbed_feat_eps_9_gamma_0.5.npy',
            'perturbed_feat_eps_10_gamma_0.5.npy',]

epsilon = [0,1,1,1,1,1,1,1,1,1,1,2,3,4,5,6,7,8,9,10]
gamma = ['00','01','02','03','04','05','06','07','08','09','10','05','05','05','05','05','05','05','05','05',]

edge_priv_path = '../Datasets/MIR_FLICKR/pairs/mir_priv.pairs'
edge_generate_path = '../Datasets/MIR_FLICKR/pairs/mir_priv_random.pairs'
mask_tr = np.load('../Datasets/MIR_FLICKR/feat/mask_tr.npy')
mask_va = np.load('../Datasets/MIR_FLICKR/feat/mask_va.npy')
mask_te = np.load('../Datasets/MIR_FLICKR/feat/mask_te.npy')
image_df = pd.read_csv('../Datasets/MIR_FLICKR/mir.csv')

neg, pos = np.bincount(np.squeeze(image_df[label].to_numpy()))
total = neg + pos
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

edge_list = []
file = open(edge_priv_path, 'r')
lines = file.readlines()
for line in lines:
    temp = line.split()
    if (int(temp[-1]) == 0):
        edge_list.append((int(temp[0]), int(temp[1])))
file = open(edge_generate_path, 'r')
lines = file.readlines()
for line in lines:
    temp = line.split()
    edge_list.append((int(temp[0]), int(temp[-1])))
adj_matrix = adj_from_edge_list(edge_list, num_nodes)
adj_matrix = laplacian_matrix(adj_matrix)

dataset = Citation('cora', normalize_x=True, transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])
weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (mask_tr, mask_va, mask_te)
)

sA = scipy.sparse.csr_matrix(adj_matrix)
dataset[0].a = sA
dataset[0].y = image_df[[label]]

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


print("Shape of Adjacency matrix", adj_matrix.shape)

def base_model():
    x_in = tf.keras.Input(shape=(num_feats,))
    a_in = tf.keras.Input((num_nodes,), sparse=True)
    x = GCNConv(channels,
                activation='relu',
                use_bias=True,
                kernel_regularizer = tf.keras.regularizers.l2(l2_reg))([x_in, a_in])
    x = tf.keras.layers.Dropout(dropout)(x)
    x = GCNConv(1,
                activation='sigmoid',
                use_bias=True)([x, a_in])
    model = tf.keras.Model(inputs=[x_in, a_in], outputs= x)
    return model

result = {}

for i in tqdm(range(20)):
    print("File:", file_path[i])
    feat_matrix = np.load(data_path + file_path[i])
    dataset[0].x = feat_matrix
    loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
    loader_va = SingleLoader(dataset, sample_weights=mask_va)
    loader_te = SingleLoader(dataset, sample_weights=mask_te)
    eps = epsilon[i]
    gam = gamma[i]
    for run in range(num_run):
        model = base_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics= [tf.keras.metrics.AUC(), f1_metric, tf.keras.metrics.BinaryAccuracy()],
        )
        history = model.fit(
            loader_tr.load(),
            steps_per_epoch=loader_tr.steps_per_epoch,
            validation_data=loader_va.load(),
            validation_steps=loader_va.steps_per_epoch,
            epochs=epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)],
            class_weight=class_weight,
            verbose=2
        )
        eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
        print("Run:", run+1, eval_results)
        result['epsilon_{}_gamma_{}_run_{}'.format(eps,gam,run+1)] = eval_results
        model.save('../saved_model/edge_channel_{}_lr_{}_epsilon_{}_gamma_{}_run_{}.h5'.format(channels,learning_rate,eps,gam,run))

for key in result:
    print(key, result[key])


