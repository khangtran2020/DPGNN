import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import pandas as pd
from scipy import sparse as sp


class PPIDataset(DGLDataset):
    def __init__(self, eps_feat, eps_edge, mode, p_rate):
        self.eps_feat = eps_feat
        self.mode = mode
        self.org_data = dgl.data.PPIDataset(mode=mode)

        # epsilon edge
        if eps_edge == '0.1':
            self.eps_edge = '01'
        elif eps_edge == '0.2':
            self.eps_edge = '02'
        elif eps_edge == '0.4':
            self.eps_edge = '04'
        elif eps_edge == '0.6':
            self.eps_edge = '06'
        elif eps_edge == '0.8':
            self.eps_edge = '08'
        elif eps_edge == '1.0':
            self.eps_edge = '1'
        elif eps_edge == '2.0':
            self.eps_edge = '2'

        # private edge rate
        if p_rate == '0.05':
            self.p_rate = '005'
        elif p_rate == '0.1':
            self.p_rate = '01'
        elif p_rate == '0.2':
            self.p_rate = '02'
        elif p_rate == '0.3':
            self.p_rate = '03'

        if mode == 'train':
            self.edge_org = [
                'ppi_0_{}.pairs'.format(self.p_rate),
                'ppi_1_{}.pairs'.format(self.p_rate),
                'ppi_2_{}.pairs'.format(self.p_rate),
                'ppi_3_{}.pairs'.format(self.p_rate),
                'ppi_4_{}.pairs'.format(self.p_rate),
                'ppi_5_{}.pairs'.format(self.p_rate),
                'ppi_6_{}.pairs'.format(self.p_rate),
                'ppi_7_{}.pairs'.format(self.p_rate),
                'ppi_8_{}.pairs'.format(self.p_rate),
                'ppi_9_{}.pairs'.format(self.p_rate),
                'ppi_10_{}.pairs'.format(self.p_rate),
                'ppi_11_{}.pairs'.format(self.p_rate),
                'ppi_12_{}.pairs'.format(self.p_rate),
                'ppi_13_{}.pairs'.format(self.p_rate),
                'ppi_14_{}.pairs'.format(self.p_rate),
                'ppi_15_{}.pairs'.format(self.p_rate),
                'ppi_16_{}.pairs'.format(self.p_rate),
                'ppi_17_{}.pairs'.format(self.p_rate),
                'ppi_18_{}.pairs'.format(self.p_rate),
                'ppi_19_{}.pairs'.format(self.p_rate),
            ]
        elif mode == 'valid':
            self.edge_org = [
                'ppi_0_{}_val.pairs'.format(self.p_rate),
                'ppi_1_{}_val.pairs'.format(self.p_rate)
            ]
        else:
            self.edge_org = [
                'ppi_0_{}_test.pairs'.format(self.p_rate),
                'ppi_1_{}_test.pairs'.format(self.p_rate)
            ]

        # mode of dataset
        if mode == 'train':
            self.num_graphs = 20
        elif mode == 'valid':
            self.num_graphs = 2
        else:
            self.num_graphs = 2

        self.data_name = 'ppi'
        self.feat_dir = 'Data/PPI/feats/'
        self.pair_dir = 'Data/PPI/pairs/'

        if mode == 'train':
            self.feat_list = [
                'perturbppi_0_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_1_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_2_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_3_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_4_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_5_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_6_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_7_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_8_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_9_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_10_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_11_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_12_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_13_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_14_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_15_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_16_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_17_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_18_feateps_{}.npy'.format(self.eps_feat),
                'perturbppi_19_feateps_{}.npy'.format(self.eps_feat),
            ]
        elif mode == 'valid':
            self.feat_list = [
                'perturbppi_0_feat_valeps_{}.npy'.format(self.eps_feat),
                'perturbppi_1_feat_valeps_{}.npy'.format(self.eps_feat)
            ]
        else:
            self.feat_list = [
                'perturbppi_0_feat_testeps_{}.npy'.format(self.eps_feat),
                'perturbppi_1_feat_testeps_{}.npy'.format(self.eps_feat)
            ]

        if mode == 'train':
            self.pair_list = [
                'ppi_0_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_1_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_2_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_3_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_4_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_5_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_6_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_7_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_8_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_9_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_10_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_11_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_12_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_13_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_14_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_15_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_16_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_17_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_18_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_19_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
            ]
        elif mode == 'valid':
            self.pair_list = [
                'ppi_0_{}_val_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_1_{}_val_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
            ]
        else:
            self.pair_list = [
                'ppi_0_{}_test_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_1_{}_test_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
            ]
        super().__init__(name='ppi')

    def process(self, ):
        self.graphs = []
        self.num_labels = self.org_data.num_labels
        for i in range(self.num_graphs):
            node_data = np.load(self.feat_dir + self.feat_list[i]).astype(np.float32)
            edge_src = []
            edge_dst = []
            file = open(self.pair_dir + self.edge_org[i], 'r')
            for line in file.readlines():
                temp = line.split()
                if int(temp[-1]) == 0:
                    edge_src.append(int(temp[0]))
                    edge_dst.append(int(temp[1]))
                    edge_src.append(int(temp[1]))
                    edge_dst.append(int(temp[0]))
            file.close()
            file = open(self.pair_dir + self.pair_list[i], 'r')
            for line in file.readlines():
                temp = line.split()
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
            file.close()
            edge_src = np.array(edge_src)
            edge_dst = np.array(edge_dst)
            node_label = self.org_data[i].ndata['label']
            n_nodes = node_data.shape[0]
            graph = dgl.graph((edge_src, edge_dst), num_nodes=n_nodes)
            graph.ndata['feat'] = torch.from_numpy(node_data)
            graph.ndata['label'] = node_label
            self.graphs.append(graph)

    def __getitem__(self, item):
        return self.graphs

    def __len__(self):
        return self.num_graphs


class PPIEdgeDataset(DGLDataset):
    def __init__(self, eps_edge, mode, p_rate):
        self.mode = mode
        self.org_data = dgl.data.PPIDataset(mode=mode)

        # epsilon edge
        if eps_edge == '0.1':
            self.eps_edge = '01'
        elif eps_edge == '0.2':
            self.eps_edge = '02'
        elif eps_edge == '0.4':
            self.eps_edge = '04'
        elif eps_edge == '0.6':
            self.eps_edge = '06'
        elif eps_edge == '0.8':
            self.eps_edge = '08'
        elif eps_edge == '1.0':
            self.eps_edge = '1'
        elif eps_edge == '2.0':
            self.eps_edge = '2'

        # private edge rate
        if p_rate == '0.05':
            self.p_rate = '005'
        elif p_rate == '0.1':
            self.p_rate = '01'
        elif p_rate == '0.2':
            self.p_rate = '02'
        elif p_rate == '0.3':
            self.p_rate = '03'

        if mode == 'train':
            self.edge_org = [
                'ppi_0_{}.pairs'.format(self.p_rate),
                'ppi_1_{}.pairs'.format(self.p_rate),
                'ppi_2_{}.pairs'.format(self.p_rate),
                'ppi_3_{}.pairs'.format(self.p_rate),
                'ppi_4_{}.pairs'.format(self.p_rate),
                'ppi_5_{}.pairs'.format(self.p_rate),
                'ppi_6_{}.pairs'.format(self.p_rate),
                'ppi_7_{}.pairs'.format(self.p_rate),
                'ppi_8_{}.pairs'.format(self.p_rate),
                'ppi_9_{}.pairs'.format(self.p_rate),
                'ppi_10_{}.pairs'.format(self.p_rate),
                'ppi_11_{}.pairs'.format(self.p_rate),
                'ppi_12_{}.pairs'.format(self.p_rate),
                'ppi_13_{}.pairs'.format(self.p_rate),
                'ppi_14_{}.pairs'.format(self.p_rate),
                'ppi_15_{}.pairs'.format(self.p_rate),
                'ppi_16_{}.pairs'.format(self.p_rate),
                'ppi_17_{}.pairs'.format(self.p_rate),
                'ppi_18_{}.pairs'.format(self.p_rate),
                'ppi_19_{}.pairs'.format(self.p_rate),
            ]
        elif mode == 'valid':
            self.edge_org = [
                'ppi_0_{}_val.pairs'.format(self.p_rate),
                'ppi_1_{}_val.pairs'.format(self.p_rate)
            ]
        else:
            self.edge_org = [
                'ppi_0_{}_test.pairs'.format(self.p_rate),
                'ppi_1_{}_test.pairs'.format(self.p_rate)
            ]

        # mode of dataset
        if mode == 'train':
            self.num_graphs = 20
        elif mode == 'valid':
            self.num_graphs = 2
        else:
            self.num_graphs = 2

        self.data_name = 'ppi'
        self.feat_dir = 'Data/PPI/feats/'
        self.pair_dir = 'Data/PPI/pairs/'

        if mode == 'train':
            self.feat_list = [
                'ppi_0_feat.npy',
                'ppi_1_feat.npy',
                'ppi_2_feat.npy',
                'ppi_3_feat.npy',
                'ppi_4_feat.npy',
                'ppi_5_feat.npy',
                'ppi_6_feat.npy',
                'ppi_7_feat.npy',
                'ppi_8_feat.npy',
                'ppi_9_feat.npy',
                'ppi_10_feat.npy',
                'ppi_11_feat.npy',
                'ppi_12_feat.npy',
                'ppi_13_feat.npy',
                'ppi_14_feat.npy',
                'ppi_15_feat.npy',
                'ppi_16_feat.npy',
                'ppi_17_feat.npy',
                'ppi_18_feat.npy',
                'ppi_19_feat.npy',
            ]
        elif mode == 'valid':
            self.feat_list = [
                'ppi_0_feat_val.npy',
                'ppi_1_feat_val.npy'
            ]
        else:
            self.feat_list = [
                'ppi_0_feat_test.npy',
                'ppi_1_feat_test.npy'
            ]

        if mode == 'train':
            self.pair_list = [
                'ppi_0_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_1_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_2_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_3_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_4_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_5_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_6_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_7_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_8_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_9_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_10_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_11_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_12_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_13_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_14_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_15_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_16_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_17_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_18_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_19_{}_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
            ]
        elif mode == 'valid':
            self.pair_list = [
                'ppi_0_{}_val_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_1_{}_val_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
            ]
        else:
            self.pair_list = [
                'ppi_0_{}_test_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
                'ppi_1_{}_test_eps_{}.pairs'.format(self.p_rate, self.eps_edge),
            ]
        super().__init__(name='ppi')

    def process(self, ):
        self.graphs = []
        self.num_labels = self.org_data.num_labels
        for i in range(self.num_graphs):
            node_data = np.load(self.feat_dir + self.feat_list[i]).astype(np.float32)
            edge_src = []
            edge_dst = []
            file = open(self.pair_dir + self.edge_org[i], 'r')
            for line in file.readlines():
                temp = line.split()
                if int(temp[-1]) == 0:
                    edge_src.append(int(temp[0]))
                    edge_dst.append(int(temp[1]))
                    edge_src.append(int(temp[1]))
                    edge_dst.append(int(temp[0]))
            file.close()
            file = open(self.pair_dir + self.pair_list[i], 'r')
            for line in file.readlines():
                temp = line.split()
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
            file.close()
            edge_src = np.array(edge_src)
            edge_dst = np.array(edge_dst)
            node_label = self.org_data[i].ndata['label']
            n_nodes = node_data.shape[0]
            graph = dgl.graph((edge_src, edge_dst), num_nodes=n_nodes)
            graph.ndata['feat'] = torch.from_numpy(node_data)
            graph.ndata['label'] = node_label
            self.graphs.append(graph)

    def __getitem__(self, item):
        return self.graphs

    def __len__(self):
        return self.num_graphs


class PPIKDDDataset(DGLDataset):
    def __init__(self, eps_edge, mode):
        self.eps_edge = eps_edge
        self.mode = mode
        self.org_data = dgl.data.PPIDataset(mode=mode)

        # mode of dataset
        if mode == 'train':
            self.num_graphs = 20
        elif mode == 'valid':
            self.num_graphs = 2
        else:
            self.num_graphs = 2

        self.data_name = 'ppi'
        self.feat_dir = 'Data/PPI/feats/'
        self.pair_dir = 'Data/PPI/pairs/'

        if mode == 'train':
            self.feat_list = [
                'ppi_0_feat.npy',
                'ppi_1_feat.npy',
                'ppi_2_feat.npy',
                'ppi_3_feat.npy',
                'ppi_4_feat.npy',
                'ppi_5_feat.npy',
                'ppi_6_feat.npy',
                'ppi_7_feat.npy',
                'ppi_8_feat.npy',
                'ppi_9_feat.npy',
                'ppi_10_feat.npy',
                'ppi_11_feat.npy',
                'ppi_12_feat.npy',
                'ppi_13_feat.npy',
                'ppi_14_feat.npy',
                'ppi_15_feat.npy',
                'ppi_16_feat.npy',
                'ppi_17_feat.npy',
                'ppi_18_feat.npy',
                'ppi_19_feat.npy'
            ]
        elif mode == 'valid':
            self.feat_list = [
                'ppi_0_feat_val.npy',
                'ppi_1_feat_val.npy'
            ]
        else:
            self.feat_list = [
                'ppi_0_feat_test.npy',
                'ppi_1_feat_test.npy'
            ]

        if mode == 'train':
            self.pair_list = [
                'ppi_0_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_1_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_2_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_3_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_4_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_5_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_6_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_7_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_8_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_9_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_10_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_11_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_12_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_13_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_14_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_15_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_16_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_17_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_18_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_19_kdd_{}.pairs'.format(self.eps_edge),
            ]
        elif mode == 'valid':
            self.pair_list = [
                'ppi_0_val_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_1_val_kdd_{}.pairs'.format(self.eps_edge),
            ]
        else:
            self.pair_list = [
                'ppi_0_test_kdd_{}.pairs'.format(self.eps_edge),
                'ppi_1_test_kdd_{}.pairs'.format(self.eps_edge),
            ]
        super().__init__(name='ppi')

    def process(self, ):
        self.graphs = []
        self.num_labels = self.org_data.num_labels
        for i in range(self.num_graphs):
            node_data = np.load(self.feat_dir + self.feat_list[i]).astype(np.float32)
            edge_src = []
            edge_dst = []
            file = open(self.pair_dir + self.pair_list[i], 'r')
            for line in file.readlines():
                temp = line.split()
                edge_src.append(int(temp[0]))
                edge_dst.append(int(temp[1]))
                edge_src.append(int(temp[1]))
                edge_dst.append(int(temp[0]))
            file.close()
            edge_src = np.array(edge_src)
            edge_dst = np.array(edge_dst)
            node_label = self.org_data[i].ndata['label']
            n_nodes = node_data.shape[0]
            graph = dgl.graph((edge_src, edge_dst), num_nodes=n_nodes)
            graph.ndata['feat'] = torch.from_numpy(node_data)
            graph.ndata['label'] = node_label
            self.graphs.append(graph)

    def __getitem__(self, item):
        return self.graphs

    def __len__(self):
        return self.num_graphs
