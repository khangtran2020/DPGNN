import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import dgl
import numpy as np
from tqdm import tqdm
from Utils.DataProcessing import *
from Datasets.FlickrDataset import FlickrMIRDataset
from Models.GCN import GCN
from Trainer.Trainer import Trainer
import torch



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_channel = 128
learning_rate = 0.01
epochs = 20000
patience = 50
num_run = 5
num_feat = 2048

dataset = dgl.data.RedditDataset()
num_class = dataset.num_classes
feat_matrix = None
file_path = 'Data/REDDIT/eps2/'
for i in range(1, 234):
    f = file_path + 'reddit_perturbed_feat_eps_2.0_batch_{}.npy'.format(i)
    if i == 1:
        feat_matrix = np.load(f)
    else:
        feat_matrix = np.concatenate((feat_matrix, np.load(f)))
print(feat_matrix.shape)
feat_matrix = feat_matrix.astype(np.float32)
save_model_path = '28DEC2021/'
dataset[0].ndata['feat'] = torch.from_numpy(feat_matrix)
del(feat_matrix)

all_result = {}
avg_result = {}

temp_f1 = []

for run in range(num_run):
    print("Run {}".format(run + 1))
    name_model_to_save = save_model_path + "reddit_feat_model_eps_2_run_{}.pt".format(run+1)
    model = GCN(in_feats= dataset[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes = num_class)
    trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                name_model=name_model_to_save, device=device)
    f1 = trainer.train_reddit()
    all_result["reddit_feat_model_eps_2_run_{}".format(run+1)] = f1
    temp_f1.append(f1)
avg_result["reddit_feat_model_eps_2"] = np.mean(np.array(temp_f1))


print("=============== ALL RESULTS: ===================")
for key in all_result:
    print(key, all_result[key])

print("=============== AVG RESULTS: ===================")
for key in avg_result:
    print(key, avg_result[key])
