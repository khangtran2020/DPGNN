import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import dgl
import numpy as np
from tqdm import tqdm
from Utils.DataProcessing import *
from Models.GCN import GCNMultiLabel
from Datasets.PPIDataset import PPIDataset
from Trainer.Trainer import TrainerPPI
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
private_edge_rate = sys.argv[1]
epsilon_feat = sys.argv[3]
epsilon_edge = sys.argv[2]

# eps_feat, eps_edge, mode, p_rate 
dataset_train = PPIDataset(eps_feat=epsilon_feat, eps_edge=epsilon_edge, mode='train', p_rate=private_edge_rate, exp_mode='featedge')
dataset_val = PPIDataset(eps_feat=epsilon_feat, eps_edge=epsilon_edge, mode='valid', p_rate=private_edge_rate, exp_mode='featedge')
dataset_test = PPIDataset(eps_feat=epsilon_feat, eps_edge=epsilon_edge, mode='test', p_rate=private_edge_rate, exp_mode='featedge')
num_class = dataset_train.num_labels
save_model_path = '03JAN2022/'

all_result = {}
avg_result = {}

temp_f1 = []

# (self, num_epoch, learning_rate, patience, model, dataset_train, dataset_val, dataset_test, name_model, device, train_mode)

for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "ppi_p_rate_{}_feateps_{}_edgeeps_{}_model_run_{}.pt".format(private_edge_rate,epsilon_feat, epsilon_edge, run+1)
    model = GCNMultiLabel(in_feats= dataset_train[0][0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
    trainer = TrainerPPI(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset_train=dataset_train,
                dataset_val=dataset_val, dataset_test=dataset_test, name_model=name_model_to_save, device=device, train_mode='featedge')
    f1 = trainer.train_feat_edge()
    all_result["ppi_p_rate_{}_feateps_{}_edgeeps_{}_model_run_{}".format(private_edge_rate,epsilon_feat, epsilon_edge, run+1)] = f1
    temp_f1.append(f1)
avg_result["ppi_p_rate_{}_feateps_{}_edgeeps_{}".format(private_edge_rate,epsilon_feat, epsilon_edge)] = np.mean(np.array(temp_f1))


print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
