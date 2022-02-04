import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import pandas as pd
from scipy import sparse as sp


class FlickrMIRDataset(DGLDataset):
    def __init__(self, feat_file, edge_org, edge_generated, type_test):
        self.feat_file = feat_file
        self.edge_org = edge_org
        self.edge_generated = edge_generated
        self.type_test = type_test
        self.data_name = 'flickr'
        super().__init__(name='flickr')

    def process(self, ):
        node_data = np.load(self.feat_file).astype(np.float32)
        print("Min value is {} and max value is {}".format(np.min(node_data), np.max(node_data)))
        img_df = pd.read_csv('Data/MIR/mir.csv')
        train_mask = np.load('Data/MIR/feat/mask_tr.npy').astype(bool)
        val_mask = np.load('Data/MIR/feat/mask_va.npy').astype(bool)
        test_mask = np.load('Data/MIR/feat/mask_te.npy').astype(bool)
        edge_src = []
        edge_dst = []
        priv_edge_src = []
        priv_edge_dst = []
        total_src = []
        total_dst = []
        priv_nodes = []
        if (self.type_test == 'feat'):
            file = open(self.edge_org, 'r')
            for line in file.readlines():
                temp = line.split()
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
                total_src.append(int(temp[0]))
                total_dst.append(int(temp[1]))
                total_src.append(int(temp[1]))
                total_dst.append(int(temp[0]))
                if int(temp[-1]) == 1:
                    priv_edge_src.append(int(temp[0]))
                    priv_edge_dst.append(int(temp[1]))
                    priv_edge_src.append(int(temp[1]))
                    priv_edge_dst.append(int(temp[0]))
                    priv_nodes.append(int(temp[0]))
                    priv_nodes.append(int(temp[1]))
            file.close()
        elif (self.type_test == 'edge_base'):
            file = open(self.edge_org, 'r')
            for line in file.readlines():
                temp = line.split()
                total_src.append(int(temp[0]))
                total_dst.append(int(temp[1]))
                total_src.append(int(temp[1]))
                total_dst.append(int(temp[0]))
                if int(temp[-1]) == 1:
                    priv_edge_src.append(int(temp[0]))
                    priv_edge_dst.append(int(temp[1]))
                    priv_edge_src.append(int(temp[1]))
                    priv_edge_dst.append(int(temp[0]))
                    priv_nodes.append(int(temp[0]))
                    priv_nodes.append(int(temp[1]))
            file.close()
            file = open(self.edge_generated, 'r')
            for line in file.readlines():
                temp = line.split()
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
            file.close()
        elif (self.type_test == 'cola'):
            file = open(self.edge_org, 'r')
            for line in file.readlines():
                temp = line.split()
                total_src.append(int(temp[0]))
                total_dst.append(int(temp[1]))
                total_src.append(int(temp[1]))
                total_dst.append(int(temp[0]))
                if int(temp[-1]) == 0:
                    edge_src.append(int(temp[0]))
                    edge_dst.append(int(temp[1]))
                    edge_src.append(int(temp[1]))
                    edge_dst.append(int(temp[0]))
                else:
                    priv_edge_src.append(int(temp[0]))
                    priv_edge_dst.append(int(temp[1]))
                    priv_edge_src.append(int(temp[1]))
                    priv_edge_dst.append(int(temp[0]))
                    priv_nodes.append(int(temp[0]))
                    priv_nodes.append(int(temp[1]))
            file.close()
            file = open(self.edge_generated, 'r')
            for line in file.readlines():
                temp = line.split()
                total_src.append(int(temp[0]))
                total_dst.append(int(temp[1]))
                total_src.append(int(temp[1]))
                total_dst.append(int(temp[0]))
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
            file.close()
        edge_src = np.array(edge_src)
        edge_dst = np.array(edge_dst)
        node_label = img_df[['people']].to_numpy().astype(np.float32)
        n_nodes = node_data.shape[0]
        self.priv_edge_adj = sp.csr_matrix(
            (np.ones(len(priv_edge_src)), (np.array(priv_edge_src), np.array(priv_edge_dst))))
        self.priv_nodes = list(set(priv_nodes))
        self.num_classes = node_label.shape[1]
        self.num_feature = node_data.shape[1]
        self.graph = dgl.graph((edge_src, edge_dst), num_nodes=n_nodes)
        self.graph.ndata['feat'] = torch.from_numpy(node_data)
        self.graph.ndata['label'] = torch.from_numpy(node_label)
        self.graph.ndata['train_mask'] = torch.from_numpy(train_mask)
        self.graph.ndata['val_mask'] = torch.from_numpy(val_mask)
        self.graph.ndata['test_mask'] = torch.from_numpy(test_mask)

    def __getitem__(self, item):
        return self.graph

    def __len__(self):
        return 1


class FlickrNUSDataset(DGLDataset):
    def __init__(self, feat_file, feat_folder, num_batch, edge_org, edge_generated, type_test):
        self.feat_file = feat_file
        self.feat_folder = feat_folder
        self.num_batch = num_batch
        self.edge_org = edge_org
        self.edge_generated = edge_generated
        self.type_test = type_test
        self.data_name = 'nus'
        super().__init__(name='nus')

    def process(self, ):
        # features
        if ((self.type_test == 'clean') or (self.type_test == 'edge') or (self.type_test == 'edge_base')):
            node_data = np.load(self.feat_file).astype(np.float32)
            min_by_row = np.min(node_data, axis=1)
            range_by_row = np.max(node_data, axis=1) - np.min(node_data, axis=1)
            min_by_row = np.expand_dims(min_by_row, axis=-1)
            range_by_row = np.expand_dims(range_by_row, axis=-1)
            min_by_row = np.tile(min_by_row, (1, node_data.shape[1]))
            range_by_row = np.tile(range_by_row, (1, node_data.shape[1]))
            node_data = (node_data - min_by_row) / range_by_row
            del (range_by_row)
            del (min_by_row)
        else:
            node_data = None
            for i in range(self.num_batch):
                if i == 0:
                    node_data = np.load(self.feat_folder + '/batch_{}.npy'.format(i)).astype(np.float32)
                else:
                    node_data = np.concatenate(
                        (node_data, np.load(self.feat_folder + '/batch_{}.npy'.format(i)).astype(np.float32)),
                        axis=0).astype(np.float32)
        print("Min value is {} and max value is {}".format(np.min(node_data), np.max(node_data)))

        # label & mask
        img_df = pd.read_csv('Data/NUS/flickr_nus.csv')
        train_mask = np.load('Data/NUS/train_mask.npy').astype(bool)
        val_mask = np.load('Data/NUS/val_mask.npy').astype(bool)
        test_mask = np.load('Data/NUS/test_mask.npy').astype(bool)
        edge_src = []
        edge_dst = []
        priv_edge_src = []
        priv_edge_dst = []
        total_src = []
        total_dst = []
        priv_nodes = []
        if (self.type_test == 'feat') or (self.type_test == 'clean'):
            file = open(self.edge_org, 'r')
            for line in file.readlines():
                temp = line.split()
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
                total_src.append(int(temp[0]))
                total_dst.append(int(temp[1]))
                total_src.append(int(temp[1]))
                total_dst.append(int(temp[0]))
                if int(temp[-1]) == 1:
                    priv_edge_src.append(int(temp[0]))
                    priv_edge_dst.append(int(temp[1]))
                    priv_edge_src.append(int(temp[1]))
                    priv_edge_dst.append(int(temp[0]))
                    priv_nodes.append(int(temp[0]))
                    priv_nodes.append(int(temp[1]))
            file.close()
        elif (self.type_test == 'edge_base'):
            file = open(self.edge_org, 'r')
            for line in file.readlines():
                temp = line.split()
                total_src.append(int(temp[0]))
                total_dst.append(int(temp[1]))
                total_src.append(int(temp[1]))
                total_dst.append(int(temp[0]))
                if int(temp[-1]) == 1:
                    priv_edge_src.append(int(temp[0]))
                    priv_edge_dst.append(int(temp[1]))
                    priv_edge_src.append(int(temp[1]))
                    priv_edge_dst.append(int(temp[0]))
                    priv_nodes.append(int(temp[0]))
                    priv_nodes.append(int(temp[1]))
            file.close()
            file = open(self.edge_generated, 'r')
            for line in file.readlines():
                temp = line.split()
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
            file.close()
        elif ((self.type_test == 'feat_edge') or (self.type_test == 'edge')):
            file = open(self.edge_org, 'r')
            for line in file.readlines():
                temp = line.split()
                total_src.append(int(temp[0]))
                total_dst.append(int(temp[1]))
                total_src.append(int(temp[1]))
                total_dst.append(int(temp[0]))
                if int(temp[-1]) == 0:
                    edge_src.append(int(temp[0]))
                    edge_dst.append(int(temp[1]))
                    edge_src.append(int(temp[1]))
                    edge_dst.append(int(temp[0]))
                else:
                    priv_edge_src.append(int(temp[0]))
                    priv_edge_dst.append(int(temp[1]))
                    priv_edge_src.append(int(temp[1]))
                    priv_edge_dst.append(int(temp[0]))
                    priv_nodes.append(int(temp[0]))
                    priv_nodes.append(int(temp[1]))
            file.close()
            file = open(self.edge_generated, 'r')
            for line in file.readlines():
                temp = line.split()
                total_src.append(int(temp[0]))
                total_dst.append(int(temp[1]))
                total_src.append(int(temp[1]))
                total_dst.append(int(temp[0]))
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
            file.close()
        edge_src = np.array(edge_src)
        edge_dst = np.array(edge_dst)
        node_label = img_df[['person']].to_numpy().astype(np.float32)
        n_nodes = node_data.shape[0]
        self.priv_edge_adj = sp.csr_matrix(
            (np.ones(len(priv_edge_src)), (np.array(priv_edge_src), np.array(priv_edge_dst))))
        self.priv_nodes = list(set(priv_nodes))
        self.num_classes = node_label.shape[1]
        self.num_feature = node_data.shape[1]
        self.graph = dgl.graph((edge_src, edge_dst), num_nodes=n_nodes)
        self.graph.ndata['feat'] = torch.from_numpy(node_data)
        self.graph.ndata['label'] = torch.from_numpy(node_label)
        self.graph.ndata['train_mask'] = torch.from_numpy(train_mask)
        self.graph.ndata['val_mask'] = torch.from_numpy(val_mask)
        self.graph.ndata['test_mask'] = torch.from_numpy(test_mask)

    def __getitem__(self, item):
        return self.graph

    def __len__(self):
        return 1