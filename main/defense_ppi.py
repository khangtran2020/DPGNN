import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import dgl
import numpy as np
from tqdm import tqdm
from Utils.DataProcessing import *
from Attacker.Attacker import DefensePPI
from Datasets.FlickrDataset import FlickrNUSDataset
from Models.GCN import GCN
from Trainer.Trainer import Trainer, TrainerDefense
import torch
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_channel = 128
learning_rate = 0.001
epochs = 2000
patience = 50
num_run = 5
num_feat = 2048
num_class = 1
epsilon = float(sys.argv[2])
delta = 1e-4
perturb_type = sys.argv[1]

dataset_train = dgl.data.PPIDataset(mode='train')
dataset_val = dgl.data.PPIDataset(mode='valid')
dataset_test = dgl.data.PPIDataset(mode='test')
dataset_train = dgl.data.PPIDataset(mode='train')
for i in range(len(dataset_train)):
    defense = DefensePPI(graph=dataset_train[i], epsilon=epsilon, noise_type='laplace', delta=delta, perturb_type=perturb_type)
    dataset_train[i] = defense.perturb_adj()
dataset_val = dgl.data.PPIDataset(mode='valid')
for i in range(len(dataset_val)):
    defense = DefensePPI(graph=dataset_val[i], epsilon=epsilon, noise_type='laplace', delta=delta, perturb_type=perturb_type)
    dataset_val[i] = defense.perturb_adj() 
dataset_test = dgl.data.PPIDataset(mode='test')
for i in range(len(dataset_test)):
    defense = DefensePPI(graph=dataset_test[i], epsilon=epsilon, noise_type='laplace', delta=delta, perturb_type=perturb_type)
    dataset_test[i] = defense.perturb_adj() 
num_class = dataset_train.num_labels
model = GCNMultiLabel(in_feats= dataset_train[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
trainer = TrainerPPI(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset_train=dataset_train,
                dataset_val=dataset_val, dataset_test=dataset_test, name_model='22JAN2022/ppi_defense_model_lapedge_eps_{}.pt', device=device, train_mode='clean')

print("=============== START TRAINING: ===================")
f1 = trainer.train()
print("================= ALL RESULTS: ====================")
print("Result:\n* F1-score={}".format(f1))
