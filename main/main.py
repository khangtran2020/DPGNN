import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import dgl
import numpy as np
from tqdm import tqdm
from Utils.DataProcessing import *
from Datasets.FlickrDataset import FlickrMIRDataset
from Models.GCN import GCN
from Trainer.Trainer import Trainer
import torch
import sys



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_channel = 128
learning_rate = 0.001
epochs = 20000
patience = 100
num_run = 5
num_feat = 2048
num_class = 1
epsilon_feat = sys.argv[1]

data_file = 'Data/MIR/feat/'
data_edge_file = 'Data/MIR/pairs/'
save_model_path = '18JAN2022/'
org_edge_file = 'mir_priv.pairs'
# edge_file = 'mir_priv.pairs'
feat_file = [
    'perturbed_feat_eps_{}_gamma_0.1.npy'.format(epsilon_feat),
    'perturbed_feat_eps_{}_gamma_0.2.npy'.format(epsilon_feat),
    'perturbed_feat_eps_{}_gamma_0.3.npy'.format(epsilon_feat),
    'perturbed_feat_eps_{}_gamma_0.4.npy'.format(epsilon_feat),
    'perturbed_feat_eps_{}_gamma_0.5.npy'.format(epsilon_feat),
    'perturbed_feat_eps_{}_gamma_0.6.npy'.format(epsilon_feat),
    'perturbed_feat_eps_{}_gamma_0.7.npy'.format(epsilon_feat),
    'perturbed_feat_eps_{}_gamma_0.8.npy'.format(epsilon_feat),
    'perturbed_feat_eps_{}_gamma_0.9.npy'.format(epsilon_feat),
]
gamma = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

all_result = {}
avg_result = {}
i = 0
for ffile in tqdm(feat_file):
    print("Running for feat file: {}".format(ffile))
    dataset = FlickrMIRDataset(feat_file=data_file + ffile, edge_org = data_edge_file + org_edge_file, edge_generated = data_edge_file + org_edge_file, type_test = 'feat')
    temp_auc = []
    temp_f1 = []
    temp_acc = []
    for run in range(num_run):
        print("Run {}".format(run + 1))
        name_model_to_save = save_model_path + "featlevel_epsFeat_{}_gamma_{}_run_{}.pt".format(epsilon_feat, gamma[i], run+1)
        model = GCN(in_feats=dataset.num_feature, h_feats=num_channel, num_classes=dataset.num_classes)
        trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                  name_model=name_model_to_save, device=device)
        auc, f1, acc = trainer.train()
        all_result["featlevel_epsFeat_{}_gamma_{}_run_{}".format(epsilon_feat, gamma[i], run+1)] = (auc, f1, acc)
        temp_auc.append(auc)
        temp_f1.append(f1)
        temp_acc.append(acc)
    avg_result["featlevel_epsFeat_{}_gamma_{}".format(epsilon_feat, gamma[i])] = (
    np.mean(np.array(temp_auc)), np.mean(np.array(temp_f1)), np.mean(np.array(temp_acc)))
    i += 1

print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
