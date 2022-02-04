import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import keras.backend as K

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

import scipy
import matplotlib.pyplot as plt
# import seaborn as sns
import zipfile

channels = 128
dropout = 0.2          # Dropout rate for the features
l2_reg = 5e-4 / 2      # L2 regularization rate
learning_rate = 0.001   # Learning rate
epochs = 200
patience = 20
seed = 99

alpha = 0.0001
sigma = 20
q_u = 0.6

epsilon = 10
gamma = 0.5

numbit_whole = 1
numbit_frac = 18
num_run = 5
# num_feat = 500

threshold = [x*0.1 for x in range(10)]
thres = 0
label = 'people'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

feat_matrix = np.load('Datasets/MIR_FLICKR/bovw.npy')
edge_list_file = 'Datasets/MIR_FLICKR/mir.pairs'
mask_feat = np.load('Datasets/MIR_FLICKR/bovw_masked_0.npy')
shap_score = np.load('Datasets/MIR_FLICKR/shap_score_bow.npy')
image_df = pd.read_csv('Datasets/MIR_FLICKR/mir.csv')
print("Size of feature matrix:", feat_matrix.shape)
print("Size of score vector", shap_score.shape)

num_nodes = feat_matrix.shape[0]
num_feats = feat_matrix.shape[1]
num_label = image_df.shape[1] - 3
print("Number of nodes:", num_nodes)
print("Number of feats:", num_feats)
print("Number of labels:", num_label)

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# softmax function
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

# normalize for shap score and sensitive score
def normalize(x, flip):
    if flip == True:
        x = np.max(x) - x + np.min(x)
        return softmax(x)
    else:
        return softmax(x)

# combining scores
def overal_score(alpha, beta, gamma):
    return gamma*alpha + (1-gamma)*beta

# float to binary
def float_to_binary(x, m, n):
    x_abs = np.abs(x)
    x_scaled = round(x_abs * 2 ** n)
    res = '{:0{}b}'.format(x_scaled, m + n)
    # if x >= 0:
    #     res = '0' + res
    # else:
    #     res = '1' + res
    return res

# binary to float
def binary_to_float(bstr, m, n):
    # sign = bstr[0]
    bs = bstr[0:]
    res = int(bs, 2) / 2 ** n
    # if sign == 1:
    #     res = -1 * res
    return res

float_bin = lambda x: float_to_binary(x, numbit_whole, numbit_frac)
float_to_binary_vec = np.vectorize(float_bin)
bin_float = lambda x: binary_to_float(x, numbit_whole, numbit_frac)
binary_to_float_vec = np.vectorize(bin_float)

def list_edge_from_adj(adj_matrix):
    new_adj = np.array(adj_matrix > 0).astype(int)
    new_adj = new_adj - np.identity(new_adj.shape[0])
    left = np.where(new_adj == 1)[0]
    right = np.where(new_adj == 1)[1]
    list_edge = []
    for a in range(left.shape[0]):
        i = left[a]
        j = right[a]
        if ((j,i) not in list_edge) & ((i,j) not in list_edge):
            list_edge.append((i,j))
    return list_edge

def pick_private_edge(list_edge, p_rate):
    list_edge_dict = {}
    cnt = 0
    for e in list_edge:
        chose_private_edge = np.random.choice(2,1, p=[1-p_rate, p_rate])
        list_edge_dict[e] = chose_private_edge[0]
        if (chose_private_edge[0] == 1):
            cnt += 1
    return list_edge_dict, cnt

def laplacian_matrix(adj_matrix):
    new_adj = adj_matrix + np.identity(adj_matrix.shape[0])
    D = np.sum(new_adj, axis=1)
    D = np.diag(D)
    D = np.power(D,-0.5)
    D[D == np.inf] = 0
    adj = np.dot(np.dot(D,new_adj),D)
    return adj

def adj_from_edge_list(edge_list, num_node):
    new_adj = np.zeros(shape=(num_node,num_node))
    for e in edge_list:
        new_adj[e[0], e[1]] = 1
        new_adj[e[1], e[0]] = 1
    return new_adj

def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)

X = feat_matrix
y = image_df[label]
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2, stratify=y)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=seed, test_size=0.2, stratify=y_train)
print(x_train.shape, y_train.shape)
print(x_valid.shape,  y_valid.shape)
print(x_test.shape, y_test.shape)

mask_tr = np.zeros(image_df.shape[0])
for i in y_train.index:
    mask_tr[i] = 1

mask_va = np.zeros(image_df.shape[0])
for i in y_valid.index:
    mask_va[i] = 1

mask_te = np.zeros(image_df.shape[0])
for i in y_test.index:
    mask_te[i] = 1

print(mask_tr, mask_va, mask_te)

list_id = dict(zip(image_df['flickr_id'], image_df.index))
file = open(edge_list_file, 'r')
lines = file.readlines()
edge_list = []
for line in lines:
    edge_list.append((int(line.split()[0]), int(line.split()[1])))
adj_matrix = adj_from_edge_list(edge_list, num_nodes)
print(adj_matrix.shape)

print(np.max(feat_matrix), np.min(feat_matrix))

shap_score = np.abs(shap_score)
for i in range(shap_score.shape[0]):
    shap_score[i] = softmax(shap_score[i])

sens_score = np.abs(feat_matrix - mask_feat)
for i in range(sens_score.shape[0]):
    sens_score[i] = np.max(sens_score[i]) - sens_score[i] + np.min(sens_score[i])
    sens_score[i] = softmax(sens_score[i])

theta = overal_score(shap_score, sens_score, 0.5)
print(theta.shape)

r = num_feats
l = numbit_whole + numbit_frac

perturbed_feat = None
for t in tqdm(range(feat_matrix.shape[0])):
    x = float_to_binary_vec(feat_matrix[t])
    temp = "".join(i for i in x)
    # temp = temp.split()
    indx = range(len(temp))
    choices = []
    bitstring = []
    for j in indx:
        i = int(np.floor(j/l))
        thet = theta[t,i]
        eps = epsilon*thet*r
        mu = epsilon*((l+1)*r - 1)/(2*r*l*(r*l-1))
        p = sigmoid(eps*(l-j%l)/l-mu)
        choice = random.choices([0, 1], weights=[p, 1 - p], k=1)
        choices.append(choice[0])
        bitstring.append(int(temp[i]))
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
    if t == 0:
        perturbed_feat = binary_to_float_vec(np.expand_dims(np.array(temp), axis = 0))
    else:
        perturbed_feat = np.concatenate((perturbed_feat, binary_to_float_vec(np.expand_dims(np.array(temp), axis = 0))), axis=0)

# np.save('Datasets/MIR_FLICKR/perturbed_feat_eps_' + str(epsilon) + '.npy', perturbed_feat)
dataset = Citation('cora', normalize_x=True, transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])

weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (mask_tr, mask_va, mask_te)
)

labels = ['clouds', 'male', 'bird', 'dog',
          'river', 'portrait', 'baby', 'night', 
          'people', 'female', 'sea', 'tree',
          'car', 'flower']

def base_model():
    x_in = tf.keras.Input(shape=(num_feats,))
    a_in = tf.keras.Input((num_nodes,), sparse=True)
    gc_1 = GCNConv(32,
                activation='relu',
                use_bias=True)([x_in, a_in])
    do = tf.keras.layers.Dropout(dropout)(gc_1)
    gc_2 = GCNConv(1,
                activation='sigmoid',
                use_bias=True)([do, a_in])
    model = tf.keras.Model(inputs=[x_in, a_in], outputs= gc_2)
    return model

best_thres = []
auc = []
acc = []
neg, pos = np.bincount(np.squeeze(image_df[[label]]))
total = neg + pos
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
# dataset[0].x = feat_matrix
dataset[0].x = perturbed_feat
sA = scipy.sparse.csr_matrix(adj_matrix)
dataset[0].a = sA
dataset[0].y = image_df[[label]]
loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)
loader_te = SingleLoader(dataset, sample_weights=weights_te)
max_acc = 0
max_auc = 0
curr_thres = 0
lab_auc = []
for thres in threshold:
    def binary_accuracy(y_true, y_pred, threshold=thres):
        y_pred = tf.convert_to_tensor(y_pred)
        threshold = tf.cast(threshold, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
        return K.mean(tf.equal(y_true, y_pred), axis=-1)
    model = base_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics= [tf.keras.metrics.AUC(), binary_accuracy],
    )
    history = model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)],
        class_weight=class_weight,
        verbose = 0
    )
    eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    if (eval_results[2] > max_acc):
        curr_thres = thres
        max_acc = eval_results[2]
    lab_auc.append(eval_results[1])
acc.append(max_acc)
auc.append(np.mean(np.array(lab_auc)))

print('AUC:', np.mean(np.array(auc)))
print('Accuracy:', np.mean(np.array(acc)))
