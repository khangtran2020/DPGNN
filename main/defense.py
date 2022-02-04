import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import dgl
import numpy as np
from tqdm import tqdm
from Utils.DataProcessing import *
from Attacker.Attacker import Defense
from Datasets.FlickrDataset import FlickrMIRDataset
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

feat_file = 'Data/MIR/feat/resnet50_plain_feat.npy'
edge_file = 'Data/MIR/pairs/mir_priv.pairs'
dataset = FlickrMIRDataset(feat_file=feat_file, edge_org=edge_file, edge_generated=None, type_test='feat')
defense = Defense(dataset=dataset, epsilon=epsilon, noise_type='laplace', delta=delta, perturb_type=perturb_type,
                  feat_file=feat_file)
graph = defense.perturb_adj()
model = GCN(in_feats=dataset.num_feature, h_feats=num_channel, num_classes=dataset.num_classes)
trainer = TrainerDefense(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, graph=graph,
                         name_model='22JAN2022/defense_model_lapedge_eps_{}.pt'.format(epsilon), device=device)
print("=============== START TRAINING: ===================")
auc, f1, acc = trainer.train()
print("================= ALL RESULTS: ====================")
print("Result:\n* AUC = {}\n* F1-score={}\n* Acc = {}".format(auc, f1, acc))
