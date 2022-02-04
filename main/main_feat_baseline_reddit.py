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
from Datasets.FlickrDataset import FlickrNUSDataset
from Models.GCN import GCN
from Trainer.Trainer import Trainer
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
num_batch = 47
# epsilon_feat = sys.argv[1]

data_file = 'Data/REDDIT/'
save_model_path = '03JAN2022/'
mechanism = sys.argv[1]
print(mechanism)

feat_folder = [
    '{}_eps_0.01'.format(mechanism),
    '{}_eps_0.05'.format(mechanism),
    '{}_eps_0.1'.format(mechanism),
    '{}_eps_0.2'.format(mechanism),
    '{}_eps_0.4'.format(mechanism),
    '{}_eps_0.6'.format(mechanism),
    '{}_eps_0.8'.format(mechanism),
    '{}_eps_1.0'.format(mechanism),
    '{}_eps_2.0'.format(mechanism),
]
epsilon = ['0.01', '0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '1.0', '2.0']

all_result = {}
avg_result = {}
i = 0
for folder in tqdm(feat_folder):
    print("Running for folder: {}".format(folder))
    dataset = dgl.data.RedditDataset()
    num_class = dataset.num_classes
    print(num_class)
    feat_matrix = None
    file_path = 'Data/REDDIT/{}/'.format(folder)
    for i in range(num_batch):
        f = file_path + 'batch_{}.npy'.format(i)
        if i == 0:
            feat_matrix = np.load(f)
        else:
            feat_matrix = np.concatenate((feat_matrix, np.load(f)), axis=0)
    print(feat_matrix.shape)
    feat_matrix = feat_matrix.astype(np.float32)
    dataset[0].ndata['feat'] = torch.from_numpy(feat_matrix)
    del(feat_matrix)
    temp_f1 = []
    for run in range(num_run):
        print("Run {}".format(run + 1))
        name_model_to_save = save_model_path + "reddit_{}_model_eps_{}_run_{}.pt".format(mechanism,epsilon,run+1)
        model = GCN(in_feats= dataset[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
        trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                    name_model=name_model_to_save, device=device)
        f1 = trainer.train_reddit()
        all_result["reddit_{}_model_eps_{}_run_{}".format(mechanism,epsilon,run+1)] = f1
        temp_f1.append(f1)
    avg_result["reddit_{}_model_eps_{}".format(mechanism,epsilon)] = np.mean(np.array(temp_f1))
    i += 1

print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
