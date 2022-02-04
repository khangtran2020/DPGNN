import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_channel = 128
learning_rate = 0.001
epochs = 20000
patience = 50
num_run = 5
num_feat = 2048

dataset_train = dgl.data.PPIDataset(mode='train')
dataset_val = dgl.data.PPIDataset(mode='valid')
dataset_test = dgl.data.PPIDataset(mode='test')
num_class = dataset_train.num_labels
save_model_path = '30DEC2021/'

all_result = {}
avg_result = {}

temp_f1 = []

# (self, num_epoch, learning_rate, patience, model, dataset_train, dataset_val, dataset_test, name_model, device, train_mode)

for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "ppi_clean_model_run_{}.pt".format(run+1)
    model = GCNMultiLabel(in_feats= dataset_train[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
    trainer = TrainerPPI(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset_train=dataset_train,
                dataset_val=dataset_val, dataset_test=dataset_test, name_model=name_model_to_save, device=device, train_mode='clean')
    f1 = trainer.train()
    all_result["clean_model_run_{}".format(run+1)] = f1
    temp_f1.append(f1)
avg_result["clean_model_run_{}".format(run+1)] = np.mean(np.array(temp_f1))


print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
