import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
num_run = 3
num_feat = 50
epsilon = sys.argv[1]
data_path = 'Data/PPI/feats/'

train_feat = [
    'perturbppi_0_feateps_{}.npy'.format(epsilon),
    'perturbppi_1_feateps_{}.npy'.format(epsilon),
    'perturbppi_2_feateps_{}.npy'.format(epsilon),
    'perturbppi_3_feateps_{}.npy'.format(epsilon),
    'perturbppi_4_feateps_{}.npy'.format(epsilon),
    'perturbppi_5_feateps_{}.npy'.format(epsilon),
    'perturbppi_6_feateps_{}.npy'.format(epsilon),
    'perturbppi_7_feateps_{}.npy'.format(epsilon),
    'perturbppi_8_feateps_{}.npy'.format(epsilon),
    'perturbppi_9_feateps_{}.npy'.format(epsilon),
    'perturbppi_10_feateps_{}.npy'.format(epsilon),
    'perturbppi_11_feateps_{}.npy'.format(epsilon),
    'perturbppi_12_feateps_{}.npy'.format(epsilon),
    'perturbppi_13_feateps_{}.npy'.format(epsilon),
    'perturbppi_14_feateps_{}.npy'.format(epsilon),
    'perturbppi_15_feateps_{}.npy'.format(epsilon),
    'perturbppi_16_feateps_{}.npy'.format(epsilon),
    'perturbppi_17_feateps_{}.npy'.format(epsilon),
    'perturbppi_18_feateps_{}.npy'.format(epsilon),
    'perturbppi_19_feateps_{}.npy'.format(epsilon),
]

val_feat = [
    'perturbppi_0_feat_valeps_{}.npy'.format(epsilon),
    'perturbppi_1_feat_valeps_{}.npy'.format(epsilon)
]

test_feat = [
    'perturbppi_0_feat_testeps_{}.npy'.format(epsilon),
    'perturbppi_1_feat_testeps_{}.npy'.format(epsilon)
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
save_model_path = '07JAN2022/'

all_result = {}
avg_result = {}

temp_f1 = []

# (self, num_epoch, learning_rate, patience, model, dataset_train, dataset_val, dataset_test, name_model, device, train_mode)

for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "ppi_feat_eps_{}_run_{}.pt".format(epsilon,run+1)
    model = GCNMultiLabel(in_feats= dataset_train[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
    trainer = TrainerPPI(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset_train=dataset_train,
                dataset_val=dataset_val, dataset_test=dataset_test, name_model=name_model_to_save, device=device, train_mode='feat')
    f1 = trainer.train()
    all_result["ppi_feat_eps_{}_run_{}.pt".format(epsilon,run+1)] = f1
    temp_f1.append(f1)
avg_result["ppi_feat_eps_{}".format(epsilon)] = np.mean(np.array(temp_f1))


print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])

