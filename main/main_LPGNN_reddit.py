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
from Models.GCN import GCN
from Trainer.Trainer import Trainer
import torch
from LPGNN import *
import sys


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_channel = 128
learning_rate = 0.01
epochs = 20000
patience = 100
num_run = 5
num_feat = 602
epsilon_feat = float(sys.argv[1])

dataset = dgl.data.RedditDataset()
feat_matrix = dataset[0].ndata['feat']
print((torch.min(feat_matrix).numpy(),torch.max(feat_matrix).numpy()))
LPGNN = MultiBit(eps = epsilon_feat, input_range=(torch.min(feat_matrix).numpy(),torch.max(feat_matrix).numpy()))
perturbed_feat_matrix = LPGNN(x = feat_matrix)
dataset[0].ndata['feat'] = perturbed_feat_matrix
num_class = dataset.num_classes
save_model_path = '18JAN2022/'

all_result = {}
avg_result = {}

temp_f1 = []

for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "reddit_featlevel_LPGNN_eps_{}_run_{}.pt".format(epsilon_feat, run+1)
    model = GCN(in_feats= dataset[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
    trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                name_model=name_model_to_save, device=device)
    f1 = trainer.train_reddit()
    all_result["reddit_featlevel_LPGNN_eps_{}_run_{}".format(epsilon_feat, run+1)] = f1
    temp_f1.append(f1)
avg_result["reddit_featlevel_LPGNN_eps_{}".format(epsilon_feat)] = np.mean(np.array(temp_f1))


print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
