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
epsilon = sys.argv[1]
data_path = 'Data/PPI/feats/'
mechanism = sys.argv[2]

train_feat = [
    '{}_ppi_0_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_1_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_2_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_3_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_4_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_5_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_6_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_7_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_8_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_9_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_10_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_11_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_12_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_13_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_14_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_15_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_16_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_17_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_18_feat_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_19_feat_eps_{}.npy'.format(mechanism,epsilon),
]

val_feat = [
    '{}_ppi_0_feat_val_eps_{}.npy'.format(mechanism,epsilon),
    '{}_ppi_1_feat_val_eps_{}.npy'.format(mechanism,epsilon)
]

test_feat = [
    '{}_ppi_0_feat_test_eps_{}.npy'.format(mechanism,epsilon), # duchi_ppi_0_feat_test_eps_0.01.npy
    '{}_ppi_1_feat_test_eps_{}.npy'.format(mechanism,epsilon)
]

dataset_train = dgl.data.PPIDataset(mode='train')
for i in range(len(dataset_train)):
    feat_matrix = np.load(data_path+train_feat[i]).astype(np.float32)
    dataset_train[i].ndata['feat'] = torch.from_numpy(feat_matrix) 
dataset_val = dgl.data.PPIDataset(mode='valid')
for i in range(len(dataset_val)):
    feat_matrix = np.load(data_path+val_feat[i]).astype(np.float32)
    dataset_val[i].ndata['feat'] = torch.from_numpy(feat_matrix) 
dataset_test = dgl.data.PPIDataset(mode='test')
for i in range(len(dataset_test)):
    feat_matrix = np.load(data_path+test_feat[i]).astype(np.float32)
    dataset_test[i].ndata['feat'] = torch.from_numpy(feat_matrix)
num_class = dataset_train.num_labels
save_model_path = '30DEC2021/'

all_result = {}
avg_result = {}

temp_f1 = []

# (self, num_epoch, learning_rate, patience, model, dataset_train, dataset_val, dataset_test, name_model, device, train_mode)

for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "{}_ppi_feat_eps_{}_run_{}.pt".format(mechanism,epsilon,run+1)
    model = GCNMultiLabel(in_feats= dataset_train[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
    trainer = TrainerPPI(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset_train=dataset_train,
                dataset_val=dataset_val, dataset_test=dataset_test, name_model=name_model_to_save, device=device, train_mode='feat')
    f1 = trainer.train()
    all_result["{}_ppi_feat_eps_{}_run_{}.pt".format(mechanism,epsilon,run+1)] = f1
    temp_f1.append(f1)
avg_result["{}_ppi_feat_eps_{}".format(mechanism,epsilon)] = np.mean(np.array(temp_f1))


print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
