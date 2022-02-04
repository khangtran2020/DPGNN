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
epochs = 20000
patience = 50
num_feat = 2048
num_batch = 47
data_file = 'Data/REDDIT/'
save_model_path = '20JAN2022/'
mechanism = sys.argv[1]
epsilon_feat = sys.argv[2]
print(mechanism)

feat_folder = '{}_eps_{}'.format(mechanism, epsilon_feat)

all_result = {}
avg_result = {}
i = 0

dataset = dgl.data.RedditDataset()
num_class = dataset.num_classes
print(num_class)
feat_matrix = None
file_path = 'Data/REDDIT/{}/'.format(feat_folder)
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
name_model_to_save = save_model_path + "reddit_{}_model_eps_{}.pt".format(mechanism,epsilon_feat)
model = GCN(in_feats= dataset[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
            name_model=name_model_to_save, device=device)
f1 = trainer.train_reddit()
all_result["reddit_{}_model_eps_{}".format(mechanism,epsilon_feat)] = f1
temp_f1.append(f1)
avg_result["reddit_{}_model_eps_{}".format(mechanism,epsilon_feat)] = np.mean(np.array(temp_f1))


print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])


