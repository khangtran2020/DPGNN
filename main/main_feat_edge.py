import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import numpy as np
from tqdm import tqdm
from Utils.DataProcessing import *
from Datasets.FlickrDataset import FlickrMIRDataset
from Models.GCN import GCN
from Trainer.Trainer import Trainer
import torch



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_channel = 128
learning_rate = 0.001
epochs = 2000
patience = 200
num_run = 5
num_feat = 2048
num_class = 1
# epsilon_feat = '2.0'
epsilon_edge = '2.0'

data_file = 'Data/MIR/feat/'
data_edge_file = 'Data/MIR/pairs/'
save_model_path = '14DEC2021/'
org_edge_file = 'mir_priv.pairs'
edge_file = 'mir_priv_samemin_1_1.pairs'

feat_file = [
    'perturbed_feat_eps_0.01_gamma_0.5.npy',
    'perturbed_feat_eps_0.05_gamma_0.2.npy',
    'perturbed_feat_eps_0.1_gamma_0.1.npy',
    'perturbed_feat_eps_0.2_gamma_0.1.npy',
    'perturbed_feat_eps_0.4_gamma_0.1.npy',
    'perturbed_feat_eps_0.6_gamma_0.1.npy',
    'perturbed_feat_eps_0.8_gamma_0.1.npy',
    'perturbed_feat_eps_1.0_gamma_0.1.npy',
    'perturbed_feat_eps_2.0_gamma_0.2.npy',
]
eps_feat = ['0.01', '0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '1.0', '2.0']

all_result = {}
avg_result = {}
i = 0
for ffile in tqdm(feat_file):
    print("Running for feat file: {}".format(ffile))
    dataset = FlickrMIRDataset(feat_file=data_file + ffile, edge_org = data_edge_file + org_edge_file, edge_generated = data_edge_file + edge_file, type_test = 'cola')
    temp_auc = []
    temp_f1 = []
    temp_acc = []
    for run in range(num_run):
        print("Run {}".format(run + 1))
        name_model_to_save = save_model_path + "featedge_epsFeat_{}_epsEdge_{}_run_{}.pt".format(eps_feat[i], epsilon_edge, run+1)
        model = GCN(in_feats=dataset.num_feature, h_feats=num_channel, num_classes=dataset.num_classes)
        trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                  name_model=name_model_to_save, device=device)
        auc, f1, acc = trainer.train()
        all_result["featedge_epsFeat_{}_epsEdge_{}_run_{}".format(eps_feat[i], epsilon_edge, run+1)] = (auc, f1, acc)
        temp_auc.append(auc)
        temp_f1.append(f1)
        temp_acc.append(acc)
    avg_result["featedge_epsFeat_{}_epsEdge_{}".format(eps_feat[i], epsilon_edge)] = (
    np.mean(np.array(temp_auc)), np.mean(np.array(temp_f1)), np.mean(np.array(temp_acc)))
    i += 1

print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
