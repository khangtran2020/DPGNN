import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import numpy as np
from tqdm import tqdm
from Utils.DataProcessing import *
from Datasets.FlickrDataset import FlickrNUSDataset
from Models.GCN import GCN
from Trainer.Trainer import Trainer
from LPGNN import *
import torch
import sys



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_channel = 128
learning_rate = 0.001
epochs = 20000
patience = 50
num_run = 5
num_feat = 2048
num_class = 1
num_batch = 18
epsilon_feat = float(sys.argv[1])

data_file = 'Data/MIR/feat/'
data_edge_file = 'Data/MIR/pairs/'
save_model_path = '17JAN2022/'
org_edge_file = 'mir_priv.pairs'
feat_file = 'resnet50_plain_feat.npy'

all_result = {}
avg_result = {}
print("Running for feat file: {}".format(feat_file))
dataset = FlickrMIRDataset(feat_file=data_file + feat_file, edge_org = data_edge_file + org_edge_file, edge_generated = data_edge_file + org_edge_file, type_test = 'feat')
feat_matrix = dataset[0].ndata['feat']
# feat_matrix = torch.from_numpy(min_max_norm(feat_matrix))
LPGNN = MultiBit(eps = epsilon_feat, input_range=(0,34.557769775390625))
perturbed_feat_matrix = LPGNN(x = feat_matrix)
dataset[0].ndata['feat'] = perturbed_feat_matrix

temp_auc = []
temp_f1 = []
temp_acc = []
for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "featlevel_LPGNN_eps_{}_run_{}.pt".format(epsilon_feat, run+1)
    model = GCN(in_feats=dataset.num_feature, h_feats=num_channel, num_classes=dataset.num_classes)
    trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                name_model=name_model_to_save, device=device)
    auc, f1, acc = trainer.train()
    all_result["featlevel_LPGNN_eps_{}_run_{}".format(epsilon_feat, run+1)] = (auc, f1, acc)
    temp_auc.append(auc)
    temp_f1.append(f1)
    temp_acc.append(acc)
avg_result["featlevel_LPGNN_eps_{}".format(epsilon_feat)] = (
np.mean(np.array(temp_auc)), np.mean(np.array(temp_f1)), np.mean(np.array(temp_acc)))

print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
