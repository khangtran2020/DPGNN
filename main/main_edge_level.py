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
from Datasets.CoraDataset import CoraDataset
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
p_rate = '03'

data_file = 'Data/MIR/feat/resnet50_plain_feat.npy'
data_edge_file = 'Data/Cora/'
save_model_path = '29NOV2021/'
org_edge_file = 'cora_{}.pairs'.format(p_rate)
edge_file = [
    'cora_{}_ep_01.pairs'.format(p_rate),
    'cora_{}_ep_02.pairs'.format(p_rate),
    'cora_{}_ep_04.pairs'.format(p_rate),
    'cora_{}_ep_06.pairs'.format(p_rate),
    'cora_{}_ep_08.pairs'.format(p_rate),
    'cora_{}_ep_1.pairs'.format(p_rate),
    'cora_{}_ep_2.pairs'.format(p_rate),
]
eps = ['0.1', '0.2', '0.4', '0.6', '0.8', '1.0', '2.0']

all_result = {}
avg_result = {}
i = 0
for efile in tqdm(edge_file):
    print("Running for feat file: {}".format(efile))
    dataset = CoraDataset(edge_org = data_edge_file + org_edge_file, edge_generated = data_edge_file + efile, type_test = 'cola')
    # temp_auc = []
    # temp_f1 = []
    temp_acc = []
    for run in range(num_run):
        print("Run {}".format(run + 1))
        name_model_to_save = save_model_path + "edgelevel_epsEdge_{}_run_{}.pt".format(eps[i], run+1)
        model = GCN(in_feats=dataset.num_feature, h_feats=num_channel, num_classes=dataset.num_classes)
        trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                  name_model=name_model_to_save, device=device)
        # auc, f1, acc = trainer.train_cora()
        acc = trainer.train_cora()
        # all_result["edgelevel_epsEdge_{}_run_{}".format(eps[i], run+1)] = (auc, f1, acc)
        all_result["edgelevel_epsEdge_{}_run_{}".format(eps[i], run+1)] = acc
        # temp_auc.append(auc)
        # temp_f1.append(f1)
        temp_acc.append(acc)
    # avg_result["edgelevel_epsEdge_{}".format(eps[i])] = (np.mean(np.array(temp_auc)), np.mean(np.array(temp_f1)), np.mean(np.array(temp_acc)))
    avg_result["edgelevel_epsEdge_{}".format(eps[i])] = np.mean(np.array(temp_acc))
    i += 1

print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
