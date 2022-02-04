from collections import defaultdict
import numpy as np
from sklearn import metrics
import time
from tqdm import tqdm
import torch
import dgl
import scipy.sparse as sp
from Utils.DataProcessing import get_noise
import pandas as pd

class Attacker:
    def __init__(self, dataset, model, n_samples, influence):
        self.dataset = dataset
        self.model = model
        self.graph = self.dataset[0]
        self.graph = dgl.add_self_loop(self.graph)
        self.n_node = self.graph.ndata['feat'].shape[0]
        self.adj = self.graph.adj(scipy_fmt='csr')
        self.features = self.graph.ndata['feat']
        self.n_samples = n_samples
        self.influence = influence
        np.random.seed(1608)
        # print(self.adj.shape, self.adj.indices, self.adj.indptr)

    def get_gradient_eps(self, u, v):
        pert_1 = torch.zeros_like(self.features)
        pert_1[v] = self.features[v] * self.influence
        grad = (self.model(self.graph, self.features + pert_1).detach() -
                self.model(self.graph, self.features).detach()) / self.influence

        return grad[u]

    def get_gradient_eps_mat(self, v):
        pert_1 = torch.zeros_like(self.features)
        pert_1[v] = self.features[v] * self.influence
        grad = (self.model(self.graph, self.features + pert_1).detach() -
                self.model(self.graph, self.features).detach()) / self.influence
        return grad

    def link_prediction_attack_efficient(self):
        norm_exist = []
        norm_nonexist = []

        t = time.time()

        # 2. compute influence value for all pairs of nodes
        influence_val = np.zeros((self.n_samples, self.n_samples))

        with torch.no_grad():

            for i in tqdm(range(self.n_samples)):
                u = self.test_nodes[i]
                grad_mat = self.get_gradient_eps_mat(u)

                for j in range(self.n_samples):
                    v = self.test_nodes[j]

                    grad_vec = grad_mat[v]

                    influence_val[i][j] = grad_vec.norm().item()

            print(f'time for predicting edges: {time.time() - t}')

        node2ind = {node: i for i, node in enumerate(self.test_nodes)}

        for u, v in self.exist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_exist.append(influence_val[j][i])

        for u, v in self.nonexist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_nonexist.append(influence_val[j][i])

        self.compute_and_save(norm_exist, norm_nonexist)

    def compute_and_save(self, norm_exist, norm_nonexist):
        y = [1] * len(norm_exist) + [0] * len(norm_nonexist)
        pred = norm_exist + norm_nonexist
        print('number of prediction:', len(pred))

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        print('auc =', metrics.auc(fpr, tpr))

        precision, recall, thresholds_2 = metrics.precision_recall_curve(y, pred)
        print('ap =', metrics.average_precision_score(y, pred))

        folder_name = 'SavedModel/'
        filename = 'attack_result.pt'
        torch.save({
            'auc': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'pr': {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds_2
            },
            'result': {
                'y': y,
                'pred': pred,
            }
        }, folder_name + filename)
        print(f'attack results saved to: {filename}')

    def construct_private_edge_set(self):
        indices = self.dataset.priv_edge_adj.indices
        indptr = self.dataset.priv_edge_adj.indptr
        n_nodes = len(self.dataset.priv_nodes)
        indice_all = range(n_nodes)
        print('#indice =', len(indice_all), len(self.dataset.priv_nodes))
        nodes = np.random.choice(indice_all, self.n_samples, replace=False)  # choose from low degree nodes
        self.test_nodes = [self.dataset.priv_nodes[i] for i in nodes]
        self.exist_edges, self.nonexist_edges = self._get_edge_sets_among_nodes(indices=indices, indptr=indptr,
                                                                                     nodes=self.test_nodes)

    def construct_edge_sets_from_random_subgraph(self):
        indices = self.adj.indices
        indptr = self.adj.indptr
        n_nodes = self.adj.shape[0]
        indice_all = range(n_nodes)
        print('#indice =', len(indice_all))
        nodes = np.random.choice(indice_all, self.n_samples, replace=False)  # choose from low degree nodes
        self.test_nodes = nodes
        self.exist_edges, self.nonexist_edges = self._get_edge_sets_among_nodes(indices=indices, indptr=indptr,
                                                                                     nodes=nodes)

    def _get_edge_sets_among_nodes(self, indices, indptr, nodes):
        # construct edge list for each node
        dic = defaultdict(list)

        for u in nodes:
            begg, endd = indptr[u: u + 2]
            dic[u] = indices[begg: endd]

        n_nodes = len(nodes)
        edge_set = []
        nonedge_set = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                u, v = nodes[i], nodes[j]
                if v in dic[u]:
                    edge_set.append((u, v))
                else:
                    nonedge_set.append((u, v))

        index = np.arange(len(nonedge_set))
        index = np.random.choice(index, len(edge_set), replace=False)
        print(len(index))
        reduce_nonedge_set = [nonedge_set[i] for i in index]
        print('#nodes =', len(nodes))
        print('#edges_set =', len(edge_set))
        print('#nonedge_set =', len(reduce_nonedge_set))
        return edge_set, reduce_nonedge_set

class Defense:
    def __init__(self, dataset, epsilon, noise_type, delta, perturb_type, feat_file):
        self.dataset = dataset
        self.feat_file = feat_file
        self.graph = self.dataset[0]
        self.graph = dgl.add_self_loop(self.graph)
        self.n_node = self.graph.ndata['feat'].shape[0]
        self.adj = self.graph.adj(scipy_fmt='csr')
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.perturb_type = perturb_type
        self.delta = delta
        np.random.seed(1608)

    def perturb_adj(self):
        if self.perturb_type == 'discrete':
            new_adj = self.perturb_adj_discrete(self.adj)
        else:
            new_adj = self.perturb_adj_continuous(self.adj)
        neg_u, neg_v = np.where(new_adj.todense() != 0)
        new_graph = dgl.graph((neg_u, neg_v), num_nodes=self.n_node)
        # node_data = np.load(self.feat_file)
        # img_df = pd.read_csv('Data/MIR/mir.csv')
        # node_label = img_df[['people']].to_numpy().astype(np.float32)
        # train_mask = np.load('Data/MIR/feat/mask_tr.npy').astype(bool)
        # val_mask = np.load('Data/MIR/feat/mask_va.npy').astype(bool)
        # test_mask = np.load('Data/MIR/feat/mask_te.npy').astype(bool)
        new_graph.ndata['feat'] = self.graph.ndata['feat']
        new_graph.ndata['label'] = self.graph.ndata['label']
        new_graph.ndata['train_mask'] = self.graph.ndata['train_mask']
        new_graph.ndata['val_mask'] = self.graph.ndata['val_mask']
        new_graph.ndata['test_mask'] = self.graph.ndata['test_mask']
        return new_graph

    def perturb_adj_continuous(self, adj):
        self.n_nodes = adj.shape[0]
        n_edges = len(adj.data) // 2

        N = self.n_nodes
        t = time.time()

        A = sp.tril(adj, k=-1)
        print('getting the lower triangle of adj matrix done!')

        eps_1 = self.epsilon * 0.01
        eps_2 = self.epsilon - eps_1
        noise = get_noise(noise_type=self.noise_type, size = (N, N), seed = 1608,
                        eps=eps_2, delta=self.delta, sensitivity=1)
        noise *= np.tri(*noise.shape, k=-1, dtype=np.bool)
        print(f'generating noise done using {time.time() - t} secs!')

        A += noise
        print(f'adding noise to the adj matrix done!')

        t = time.time()
        n_edges_keep = n_edges + int(
            get_noise(noise_type=self.noise_type, size=1, seed= 1608,
                    eps=eps_1, delta=self.delta, sensitivity=1)[0])
        print(f'edge number from {n_edges} to {n_edges_keep}')

        t = time.time()
        a_r = A.A.ravel()

        n_splits = 50
        len_h = len(a_r) // n_splits
        ind_list = []
        for i in tqdm(range(n_splits - 1)):
            ind = np.argpartition(a_r[len_h*i:len_h*(i+1)], -n_edges_keep)[-n_edges_keep:]
            ind_list.append(ind + len_h * i)

        ind = np.argpartition(a_r[len_h*(n_splits-1):], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * (n_splits - 1))

        ind_subset = np.hstack(ind_list)
        a_subset = a_r[ind_subset]
        ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

        row_idx = []
        col_idx = []
        for idx in ind:
            idx = ind_subset[idx]
            row_idx.append(idx // N)
            col_idx.append(idx % N)
            assert(col_idx < row_idx)
        data_idx = np.ones(n_edges_keep, dtype=np.int32)
        print(f'data preparation done using {time.time() - t} secs!')

        mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
        return mat + mat.T

    def perturb_adj_discrete(self, adj):
        s = 2 / (np.exp(self.epsilon) + 1)
        print(f's = {s:.4f}')
        N = adj.shape[0]

        t = time.time()
        # bernoulli = np.random.binomial(1, s, N * (N-1) // 2)
        # entry = np.where(bernoulli)[0]

        np.random.seed(1608)
        bernoulli = np.random.binomial(1, s, (N, N))
        print(f'generating perturbing vector done using {time.time() - t} secs!')
        entry = np.asarray(list(zip(*np.where(bernoulli))))

        dig_1 = np.random.binomial(1, 1/2, len(entry))
        indice_1 = entry[np.where(dig_1 == 1)[0]]
        indice_0 = entry[np.where(dig_1 == 0)[0]]

        add_mat = self.construct_sparse_mat(indice_1, N)
        minus_mat = self.construct_sparse_mat(indice_0, N)

        adj_noisy = adj + add_mat - minus_mat

        adj_noisy.data[np.where(adj_noisy.data == -1)[0]] = 0
        adj_noisy.data[np.where(adj_noisy.data == 2)[0]] = 1

        return adj_noisy

    def construct_sparse_mat(self, indice, N):
        cur_row = -1
        new_indices = []
        new_indptr = []

        for i, j in tqdm(indice):
            if i >= j:
                continue

            while i > cur_row:
                new_indptr.append(len(new_indices))
                cur_row += 1

            new_indices.append(j)

        while N > cur_row:
            new_indptr.append(len(new_indices))
            cur_row += 1

        data = np.ones(len(new_indices), dtype=np.int64)
        indices = np.asarray(new_indices, dtype=np.int64)
        indptr = np.asarray(new_indptr, dtype=np.int64)

        mat = sp.csr_matrix((data, indices, indptr), (N, N))

        return mat + mat.T

class DefensePPI:
    def __init__(self, graph, epsilon, noise_type, delta, perturb_type, feat_file):
        self.graph = graph
        self.graph = self.dataset[0]
        self.graph = dgl.add_self_loop(self.graph)
        self.n_node = self.graph.ndata['feat'].shape[0]
        self.adj = self.graph.adj(scipy_fmt='csr')
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.perturb_type = perturb_type
        self.delta = delta
        np.random.seed(1608)

    def perturb_adj(self):
        if self.perturb_type == 'discrete':
            new_adj = self.perturb_adj_discrete(self.adj)
        else:
            new_adj = self.perturb_adj_continuous(self.adj)
        neg_u, neg_v = np.where(new_adj.todense() != 0)
        new_graph = dgl.graph((neg_u, neg_v), num_nodes=self.n_node)
        new_graph.ndata['feat'] = self.graph.ndata['feat']
        new_graph.ndata['label'] = self.graph.ndata['label']
        return new_graph

    def perturb_adj_continuous(self, adj):
        self.n_nodes = adj.shape[0]
        n_edges = len(adj.data) // 2

        N = self.n_nodes
        t = time.time()

        A = sp.tril(adj, k=-1)
        print('getting the lower triangle of adj matrix done!')

        eps_1 = self.epsilon * 0.01
        eps_2 = self.epsilon - eps_1
        noise = get_noise(noise_type=self.noise_type, size = (N, N), seed = 1608,
                        eps=eps_2, delta=self.delta, sensitivity=1)
        noise *= np.tri(*noise.shape, k=-1, dtype=np.bool)
        print(f'generating noise done using {time.time() - t} secs!')

        A += noise
        print(f'adding noise to the adj matrix done!')

        t = time.time()
        n_edges_keep = n_edges + int(
            get_noise(noise_type=self.noise_type, size=1, seed= 1608,
                    eps=eps_1, delta=self.delta, sensitivity=1)[0])
        print(f'edge number from {n_edges} to {n_edges_keep}')

        t = time.time()
        a_r = A.A.ravel()

        n_splits = 50
        len_h = len(a_r) // n_splits
        ind_list = []
        for i in tqdm(range(n_splits - 1)):
            ind = np.argpartition(a_r[len_h*i:len_h*(i+1)], -n_edges_keep)[-n_edges_keep:]
            ind_list.append(ind + len_h * i)

        ind = np.argpartition(a_r[len_h*(n_splits-1):], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * (n_splits - 1))

        ind_subset = np.hstack(ind_list)
        a_subset = a_r[ind_subset]
        ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

        row_idx = []
        col_idx = []
        for idx in ind:
            idx = ind_subset[idx]
            row_idx.append(idx // N)
            col_idx.append(idx % N)
            assert(col_idx < row_idx)
        data_idx = np.ones(n_edges_keep, dtype=np.int32)
        print(f'data preparation done using {time.time() - t} secs!')

        mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
        return mat + mat.T

    def perturb_adj_discrete(self, adj):
        s = 2 / (np.exp(self.epsilon) + 1)
        print(f's = {s:.4f}')
        N = adj.shape[0]

        t = time.time()
        # bernoulli = np.random.binomial(1, s, N * (N-1) // 2)
        # entry = np.where(bernoulli)[0]

        np.random.seed(1608)
        bernoulli = np.random.binomial(1, s, (N, N))
        print(f'generating perturbing vector done using {time.time() - t} secs!')
        entry = np.asarray(list(zip(*np.where(bernoulli))))

        dig_1 = np.random.binomial(1, 1/2, len(entry))
        indice_1 = entry[np.where(dig_1 == 1)[0]]
        indice_0 = entry[np.where(dig_1 == 0)[0]]

        add_mat = self.construct_sparse_mat(indice_1, N)
        minus_mat = self.construct_sparse_mat(indice_0, N)

        adj_noisy = adj + add_mat - minus_mat

        adj_noisy.data[np.where(adj_noisy.data == -1)[0]] = 0
        adj_noisy.data[np.where(adj_noisy.data == 2)[0]] = 1

        return adj_noisy

    def construct_sparse_mat(self, indice, N):
        cur_row = -1
        new_indices = []
        new_indptr = []

        for i, j in tqdm(indice):
            if i >= j:
                continue

            while i > cur_row:
                new_indptr.append(len(new_indices))
                cur_row += 1

            new_indices.append(j)

        while N > cur_row:
            new_indptr.append(len(new_indices))
            cur_row += 1

        data = np.ones(len(new_indices), dtype=np.int64)
        indices = np.asarray(new_indices, dtype=np.int64)
        indptr = np.asarray(new_indptr, dtype=np.int64)

        mat = sp.csr_matrix((data, indices, indptr), (N, N))

        return mat + mat.T