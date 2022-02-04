import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import numpy as np
from tqdm import tqdm
from Utils.DataProcessing import *
from Datasets.FlickrDataset import FlickrNUSDataset
from Models.GCN import GCN
from Trainer.Trainer import Trainer
import torch
import sys



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_channel = 128
learning_rate = 0.001
epochs = 10000
patience = 100
num_run = 1
num_feat = 2048
num_class = 1
num_batch = 18
epsilon_edge = sys.argv[1]

data_file = 'Data/NUS/feats/nus_resnet50_plain_feat.npy'
data_edge_file = 'Data/NUS/pairs/'
save_model_path = '21JAN2022/'
org_edge_file = 'flickr_nus.pairs'
edge_file = 'flickr_nus_kdd_{}.pairs'.format(epsilon_edge)

all_result = {}
avg_result = {}
i = 0
print("Running for feat file: {}".format(org_edge_file))
dataset = FlickrNUSDataset(feat_file=data_file, feat_folder=None, num_batch=num_batch, edge_org = data_edge_file+org_edge_file, edge_generated = data_edge_file+edge_file, type_test = 'edge_base')
temp_auc = []
temp_f1 = []
temp_acc = []
for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "NUS_edge_base_eps_{}_run_{}.pt".format(epsilon_edge, run+1)
    model = GCN(in_feats=dataset.num_feature, h_feats=num_channel, num_classes=dataset.num_classes)
    trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                name_model=name_model_to_save, device=device)
    auc, f1, acc = trainer.train()
    all_result["NUS_edge_base_eps_{}_run_{}".format(epsilon_edge, run+1)] = (auc, f1, acc)
    temp_auc.append(auc)
    temp_f1.append(f1)
    temp_acc.append(acc)
# avg_result["edge_base_epsEdge_{}".format(eps[i])] = (np.mean(np.array(temp_auc)), np.mean(np.array(temp_f1)), np.mean(np.array(temp_acc)))
avg_result["NUS_edge_eps_{}".format(epsilon_edge)] = np.mean(np.array(temp_acc))

print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])

