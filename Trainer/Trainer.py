from Models import GCN
import dgl
import torch
import numpy as np
import torch.nn.functional as F
from Utils.DataProcessing import *
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class Trainer:
    def __init__(self, num_epoch, learning_rate, patience, model, dataset, name_model, device):
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.patience = patience
        self.model = model
        self.dataset = dataset
        self.graph = dataset[0]
        self.name_model = name_model
        self.graph = dgl.add_self_loop(self.graph)
        self.graph = self.graph.to(device)
        self.model = self.model.to(device)
        self.save_path = 'SavedModel/'
        self.average = 'macro'

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_auc = 0
        best_test_auc = 0
        best_test_acc = 0
        best_test_f1 = 0
        curr_val_loss = np.inf
        time_wait = 0

        features = self.graph.ndata['feat']
        labels = self.graph.ndata['label']
        train_mask = self.graph.ndata['train_mask']
        val_mask = self.graph.ndata['val_mask']
        test_mask = self.graph.ndata['test_mask']
        for e in range(self.num_epoch):
            # Forward
            logits = self.model(self.graph, features)

            # Compute prediction
            pred = logits.round()
            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.binary_cross_entropy(logits[train_mask], labels[train_mask])
            val_loss = F.binary_cross_entropy(logits[val_mask], labels[val_mask])
            # Compute auc on training/validation/test
            train_auc = roc_auc_score(y_true=labels[train_mask].cpu().numpy(),
                                      y_score=logits[train_mask].cpu().detach().numpy())
            val_auc = roc_auc_score(y_true=labels[val_mask].cpu().numpy(),
                                    y_score=logits[val_mask].cpu().detach().numpy())
            test_auc = roc_auc_score(y_true=labels[test_mask].cpu().numpy(),
                                     y_score=logits[test_mask].cpu().detach().numpy())
            test_acc = accuracy_score(y_true=np.round(labels[test_mask].cpu().numpy()),
                                      y_pred=np.round(logits[test_mask].cpu().detach().numpy()))
            test_f1 = f1_score(y_true=np.round(labels[test_mask].cpu().numpy()),
                               y_pred=np.round(logits[test_mask].cpu().detach().numpy()))

            # print(train_auc)
            # Save the best validation accuracy and the corresponding test accuracy.

            if (val_loss < curr_val_loss):
                curr_val_loss = val_loss
                time_wait = 0
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_acc = test_acc
                best_test_f1 = test_f1
                torch.save(self.model, self.save_path + self.name_model)
            else:
                time_wait += 1

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (time_wait >= self.patience):
                print(
                    'Early stopping at epoch {}, loss: {:.3f}, train auc: {:.3f}, val loss {:.3f}), val auc: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e - self.patience, loss, train_auc, val_loss, val_auc, best_val_auc, best_test_auc))
                break

            if e % 200 == 0:
                print(
                    'In epoch {}, loss: {:.3f}, train auc: {:.3f}, val loss {:.3f}), val auc: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e, loss, train_auc, val_loss, val_auc, best_val_auc, test_auc))
        return best_test_auc, best_test_f1, best_test_acc
    def train_reddit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_f1 = 0
        best_test_f1 = 0
        best_test_f1 = 0
        curr_val_loss = np.inf

        features = self.graph.ndata['feat']
        print(torch.max(features), torch.min(features))
        labels = self.graph.ndata['label']
        train_mask = self.graph.ndata['train_mask']
        val_mask = self.graph.ndata['val_mask']
        test_mask = self.graph.ndata['test_mask']
        # print(self.graph.num_edges())
        for e in range(self.num_epoch):
            # Forward
            logits = self.model(self.graph, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask])
            # Compute f1 on training/validation/test
            train_f1 = f1_score(y_true=labels[train_mask].cpu().numpy(),
                                y_pred=pred[train_mask].cpu().detach().numpy(), average='macro')
            val_f1 = f1_score(y_true=labels[val_mask].cpu().numpy(),
                              y_pred=pred[val_mask].cpu().detach().numpy(), average='macro')
            test_f1 = f1_score(y_true=labels[test_mask].cpu().numpy(),
                               y_pred=pred[test_mask].cpu().detach().numpy(), average='macro')

            # Save the best validation accuracy and the corresponding test accuracy.
            if (val_loss < curr_val_loss):
                curr_val_loss = val_loss
                time_wait = 0
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                torch.save(self.model, self.save_path + self.name_model)
            else:
                time_wait += 1

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (time_wait >= self.patience):
                print(
                    'Early stopping at epoch {}, loss: {:.3f}, train f1: {:.3f}, val loss {:.3f}, val f1: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e - self.patience, loss, train_f1, val_loss, val_f1, best_val_f1, best_test_f1))
                break

            if e % 50 == 0:
                print(
                    'In epoch {}, loss: {:.3f}, train f1: {:.3f}, val loss {:.3f}, val f1: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e, loss, train_f1, val_loss, val_f1, best_val_f1, test_f1))

        del (self.model)
        del (self.graph)
        return best_test_f1


class TrainerDefense:
    def __init__(self, num_epoch, learning_rate, patience, model, graph, name_model, device):
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.patience = patience
        self.model = model
        self.graph = graph
        self.name_model = name_model
        self.graph = dgl.add_self_loop(self.graph)
        self.graph = self.graph.to(device)
        self.model = self.model.to(device)
        self.save_path = 'SavedModel/'
        self.average = 'macro'

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_auc = 0
        best_test_auc = 0
        best_test_acc = 0
        best_test_f1 = 0
        curr_val_loss = np.inf
        time_wait = 0

        features = self.graph.ndata['feat']
        labels = self.graph.ndata['label']
        train_mask = self.graph.ndata['train_mask']
        val_mask = self.graph.ndata['val_mask']
        test_mask = self.graph.ndata['test_mask']
        for e in range(self.num_epoch):
            # Forward
            logits = self.model(self.graph, features)

            # Compute prediction
            pred = logits.round()
            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.binary_cross_entropy(logits[train_mask], labels[train_mask])
            val_loss = F.binary_cross_entropy(logits[val_mask], labels[val_mask])
            # Compute auc on training/validation/test
            train_auc = roc_auc_score(y_true=labels[train_mask].cpu().numpy(),
                                      y_score=logits[train_mask].cpu().detach().numpy())
            val_auc = roc_auc_score(y_true=labels[val_mask].cpu().numpy(),
                                    y_score=logits[val_mask].cpu().detach().numpy())
            test_auc = roc_auc_score(y_true=labels[test_mask].cpu().numpy(),
                                     y_score=logits[test_mask].cpu().detach().numpy())
            test_acc = accuracy_score(y_true=np.round(labels[test_mask].cpu().numpy()),
                                      y_pred=np.round(logits[test_mask].cpu().detach().numpy()))
            test_f1 = f1_score(y_true=np.round(labels[test_mask].cpu().numpy()),
                               y_pred=np.round(logits[test_mask].cpu().detach().numpy()))

            # print(train_auc)
            # Save the best validation accuracy and the corresponding test accuracy.

            if (val_loss < curr_val_loss):
                curr_val_loss = val_loss
                time_wait = 0
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_acc = test_acc
                best_test_f1 = test_f1
                torch.save(self.model, self.save_path + self.name_model)
            else:
                time_wait += 1

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (time_wait >= self.patience):
                print(
                    'Early stopping at epoch {}, loss: {:.3f}, train auc: {:.3f}, val loss {:.3f}), val auc: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e - self.patience, loss, train_auc, val_loss, val_auc, best_val_auc, best_test_auc))
                break

            if e % 50 == 0:
                print(
                    'In epoch {}, loss: {:.3f}, train auc: {:.3f}, val loss {:.3f}), val auc: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e, loss, train_auc, val_loss, val_auc, best_val_auc, best_test_auc))
        return best_test_auc, best_test_f1, best_test_acc


class TrainerPPI:
    def __init__(self, num_epoch, learning_rate, patience, model, dataset_train, dataset_val, dataset_test, name_model,
                 device, train_mode):
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.patience = patience
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.name_model = name_model
        self.device = device
        self.model = self.model.to(self.device)
        self.save_path = 'SavedModel/'
        self.average = 'macro'
        self.train_mode = train_mode

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_f1 = 0
        best_test_f1 = 0
        curr_val_loss = np.inf
        time_wait = 0
        # criterion = nn.BCELoss()

        if ((self.train_mode == 'clean') or (self.train_mode == 'edge')):
            for i in range(len(self.dataset_train)):
                graph = self.dataset_train[i]
                feat_matrix = graph.ndata['feat'].numpy()
                feat_matrix = min_max_norm(feat_matrix)
                self.dataset_train[i].ndata['feat'] = torch.from_numpy(feat_matrix)
                del (feat_matrix)
            for i in range(len(self.dataset_val)):
                graph = self.dataset_val[i]
                feat_matrix = graph.ndata['feat'].numpy()
                feat_matrix = min_max_norm(feat_matrix)
                self.dataset_val[i].ndata['feat'] = torch.from_numpy(feat_matrix)
                del (feat_matrix)
            for i in range(len(self.dataset_test)):
                graph = self.dataset_test[i]
                feat_matrix = graph.ndata['feat'].numpy()
                feat_matrix = min_max_norm(feat_matrix)
                self.dataset_test[i].ndata['feat'] = torch.from_numpy(feat_matrix)
                del (feat_matrix)
            print("Done preprocessing!")

        for e in range(self.num_epoch):
            loss_train = 0
            loss_val = 0
            train_f1 = 0
            val_f1 = 0
            test_f1 = 0
            # train
            for i in range(len(self.dataset_train)):
                graph = self.dataset_train[i]
                graph = dgl.add_self_loop(graph)
                graph = graph.to(self.device)
                features = graph.ndata['feat']
                labels = graph.ndata['label']
                # Forward
                logits = self.model(graph, features)
                # Compute loss
                # Note that you should only compute the losses of the nodes in the training set.
                loss = F.binary_cross_entropy(logits, labels)
                # Compute f1 on training
                pred = np.array(logits.cpu().detach().numpy() > 0.5, dtype=float)
                f1 = f1_score(y_true=labels.cpu().numpy(),
                              y_pred=pred, average='macro')
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss
                train_f1 += f1
                del (graph)
                del (features)
                del (labels)
                del (logits)
                del (loss)
            # val
            for i in range(len(self.dataset_val)):
                graph = self.dataset_val[i]
                graph = dgl.add_self_loop(graph)
                graph = graph.to(self.device)
                features = graph.ndata['feat']
                labels = graph.ndata['label']
                logits = self.model(graph, features)
                loss = F.binary_cross_entropy(logits, labels)
                # Compute f1 on training
                pred = np.array(logits.cpu().detach().numpy() > 0.5, dtype=float)
                f1 = f1_score(y_true=labels.cpu().numpy(),
                              y_pred=pred, average='macro')
                loss_val += loss
                val_f1 += f1
                del (graph)
                del (features)
                del (labels)
                del (logits)
                del (loss)
            # test
            for i in range(len(self.dataset_test)):
                graph = self.dataset_test[i]
                graph = dgl.add_self_loop(graph)
                graph = graph.to(self.device)
                features = graph.ndata['feat']
                labels = graph.ndata['label']
                logits = self.model(graph, features)
                # Compute f1 on training
                pred = np.array(logits.cpu().detach().numpy() > 0.5, dtype=float)
                f1 = f1_score(y_true=labels.cpu().numpy(),
                              y_pred=pred, average='macro')
                test_f1 += f1
                del (graph)
                del (features)
                del (labels)
                del (logits)

            loss_train = loss_train / len(self.dataset_train)
            loss_val = loss_val / len(self.dataset_val)
            train_f1 = train_f1 / len(self.dataset_train)
            val_f1 = val_f1 / len(self.dataset_val)
            test_f1 = test_f1 / len(self.dataset_test)

            if (loss_val < curr_val_loss):
                curr_val_loss = loss_val
                time_wait = 0
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                torch.save(self.model, self.save_path + self.name_model)
            else:
                time_wait += 1

            if (time_wait >= self.patience):
                print(
                    'Early stopping at epoch {}, loss: {:.3f}, train f1: {:.3f}, val loss {:.3f}, val f1: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e - self.patience, loss_train, train_f1, loss_val, val_f1, best_val_f1, best_test_f1))
                break

            if e % 50 == 0:
                print(
                    'In epoch {}, loss: {:.3f}, train f1: {:.3f}, val loss {:.3f}, val f1: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e, loss_train, train_f1, loss_val, val_f1, best_val_f1, best_test_f1))
        return best_test_f1

    def train_feat_edge(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_f1 = 0
        best_test_f1 = 0
        curr_val_loss = np.inf
        time_wait = 0
        # criterion = nn.BCELoss()

        if ((self.train_mode == 'clean') or (self.train_mode == 'edge')):
            for i in range(len(self.dataset_train)):
                graph = self.dataset_train[0][i]
                feat_matrix = graph.ndata['feat'].numpy()
                feat_matrix = min_max_norm(feat_matrix)
                self.dataset_train[0][i].ndata['feat'] = torch.from_numpy(feat_matrix)
                del (feat_matrix)
            for i in range(len(self.dataset_val)):
                graph = self.dataset_val[0][i]
                feat_matrix = graph.ndata['feat'].numpy()
                feat_matrix = min_max_norm(feat_matrix)
                self.dataset_val[0][i].ndata['feat'] = torch.from_numpy(feat_matrix)
                del (feat_matrix)
            for i in range(len(self.dataset_test)):
                graph = self.dataset_test[0][i]
                feat_matrix = graph.ndata['feat'].numpy()
                feat_matrix = min_max_norm(feat_matrix)
                self.dataset_test[0][i].ndata['feat'] = torch.from_numpy(feat_matrix)
                del (feat_matrix)
            print("Done preprocessing!")

        for e in range(self.num_epoch):
            loss_train = 0
            loss_val = 0
            train_f1 = 0
            val_f1 = 0
            test_f1 = 0
            # train
            for i in range(len(self.dataset_train)):
                graph = self.dataset_train[0][i]
                graph = dgl.add_self_loop(graph)
                graph = graph.to(self.device)
                features = graph.ndata['feat']
                labels = graph.ndata['label']
                # Forward
                logits = self.model(graph, features)
                # Compute loss
                # Note that you should only compute the losses of the nodes in the training set.
                loss = F.binary_cross_entropy(logits, labels)
                # Compute f1 on training
                pred = np.array(logits.cpu().detach().numpy() > 0.5, dtype=float)
                f1 = f1_score(y_true=labels.cpu().numpy(),
                              y_pred=pred, average='macro')
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train += loss
                train_f1 += f1
                del (graph)
                del (features)
                del (labels)
                del (logits)
                del (loss)
            # val
            for i in range(len(self.dataset_val)):
                graph = self.dataset_val[0][i]
                graph = dgl.add_self_loop(graph)
                graph = graph.to(self.device)
                features = graph.ndata['feat']
                labels = graph.ndata['label']
                logits = self.model(graph, features)
                loss = F.binary_cross_entropy(logits, labels)
                # Compute f1 on training
                pred = np.array(logits.cpu().detach().numpy() > 0.5, dtype=float)
                f1 = f1_score(y_true=labels.cpu().numpy(),
                              y_pred=pred, average='macro')
                loss_val += loss
                val_f1 += f1
                del (graph)
                del (features)
                del (labels)
                del (logits)
                del (loss)
            # test
            for i in range(len(self.dataset_test)):
                graph = self.dataset_test[0][i]
                graph = dgl.add_self_loop(graph)
                graph = graph.to(self.device)
                features = graph.ndata['feat']
                labels = graph.ndata['label']
                logits = self.model(graph, features)
                # Compute f1 on training
                pred = np.array(logits.cpu().detach().numpy() > 0.5, dtype=float)
                f1 = f1_score(y_true=labels.cpu().numpy(),
                              y_pred=pred, average='macro')
                test_f1 += f1
                del (graph)
                del (features)
                del (labels)
                del (logits)

            loss_train = loss_train / len(self.dataset_train)
            loss_val = loss_val / len(self.dataset_val)
            train_f1 = train_f1 / len(self.dataset_train)
            val_f1 = val_f1 / len(self.dataset_val)
            test_f1 = test_f1 / len(self.dataset_test)

            if (loss_val < curr_val_loss):
                curr_val_loss = loss_val
                time_wait = 0
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                torch.save(self.model, self.save_path + self.name_model)
            else:
                time_wait += 1

            if (time_wait >= self.patience):
                print(
                    'Early stopping at epoch {}, loss: {:.3f}, train f1: {:.3f}, val loss {:.3f}, val f1: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e - self.patience, loss_train, train_f1, loss_val, val_f1, best_val_f1, best_test_f1))
                break

            if e % 50 == 0:
                print(
                    'In epoch {}, loss: {:.3f}, train f1: {:.3f}, val loss {:.3f}, val f1: {:.3f} (best val {:.3f}), best test: {:.3f}'.format(
                        e, loss_train, train_f1, loss_val, val_f1, best_val_f1, best_test_f1))
        return best_test_f1