import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import numpy as np
from Attacker.Attacker import Attacker
import warnings
from Models.GCN import GCN
from Trainer.Trainer import Trainer
from Datasets.FlickrDataset import FlickrMIRDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=DeprecationWarning)

num_channel = 128
learning_rate = 0.001
epochs = 2000
patience = 50
num_run = 5
num_feat = 2048
num_class = 1

np.random.seed(1608)

feat_file = 'Data/MIR/feat/resnet50_plain_feat.npy'
edge_file = 'Data/MIR/pairs/mir_priv.pairs'
generated_edge_file = 'Data/MIR/pairs/mir_priv_samemin_01_01.pairs'

dataset = FlickrMIRDataset(feat_file=feat_file, edge_org=edge_file, edge_generated=generated_edge_file, type_test='cola')
model = GCN(in_feats=dataset[0].ndata['feat'].shape[1], h_feats=num_channel, num_classes=dataset.num_classes)
trainer = Trainer(num_epoch=epochs, learning_rate=learning_rate, patience=patience, model=model, dataset=dataset,
                  name_model='flickr_model.pt', device=device)
print("=============== START TRAINING: ===================")
acc = trainer.train()
print("================= ALL RESULTS: ====================")
print("Result: Acc = {}".format(acc))

model = torch.load('SavedModel/flickr_model.pt')
attack = Attacker(dataset=dataset, model=model, n_samples=500, influence=0.01)
attack.construct_private_edge_set()
attack.link_prediction_attack_efficient()