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

data_file = 'Data/NUS/feat/nus_resnet50_plain_feat.npy'
data_edge_file = 'Data/NUS/pairs/flickr_nus.pairs'
feat_folder = 'Data/NUS/feat/'
save_model_path = '30DEC2021/'

all_result = {}
avg_result = {}
print("Running for feat file: {}".format(data_file.split('/')[-1]))
dataset = FlickrNUSDataset(feat_file=data_file, feat_folder=None, num_batch=None, edge_org = data_edge_file, edge_generated = None, type_test = 'clean')
temp_auc = []
temp_f1 = []
temp_acc = []
for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "clean_model_nus_run_{}.pt".format(run+1)
    model = GCN(in_feats=dataset.num_feature, h_feats=num_channel, num_classes=dataset.num_classes)
    trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                name_model=name_model_to_save, device=device)
    auc, f1, acc = trainer.train()
    all_result["clean_model_nus_run_{}".format(run+1)] = (auc, f1, acc)
    temp_auc.append(auc)
    temp_f1.append(f1)
    temp_acc.append(acc)
avg_result["clean_model_nus"] = (np.mean(np.array(temp_auc)), np.mean(np.array(temp_f1)), np.mean(np.array(temp_acc)))


print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
