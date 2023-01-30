import itertools

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

from model.rootModel import Trainer


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, name='gcn'):
        super(GNNEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if name == 'gcn':
            self.conv_layers = nn.ModuleList([
                GCNConv(self.in_channels, self.hidden_channels),
                GCNConv(self.hidden_channels, self.hidden_channels)
            ])
        elif name == 'gat':
            self.conv_layers = nn.ModuleList([
                GATConv(self.in_channels, self.hidden_channels),
                GATConv(self.hidden_channels, self.hidden_channels)
            ])
        elif name == 'gsage':
            self.conv_layers = nn.ModuleList([
                SAGEConv(self.in_channels, self.hidden_channels),
                SAGEConv(self.hidden_channels, self.hidden_channels)
            ])
        elif name == 'gin':
            self.conv_layers = nn.ModuleList([
                GINConv(nn.Linear(self.in_channels, self.hidden_channels)),
                GINConv(nn.Linear(self.hidden_channels, self.hidden_channels))
            ])

        self.prelu = nn.PReLU(self.hidden_channels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, edge_index):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = self.prelu(x)
                x = self.dropout(x)
        return x


def construct_gnn(name, encoder_dim, num_features, num_classes, device):
    encoder = GNNEncoder(num_features, encoder_dim, name).to(device)
    cls_model = nn.Sequential(
        nn.Linear(encoder_dim, num_classes),
    ).to(device)

    models = [encoder, cls_model]
    return models


class GNNTrainer(Trainer):
    def __init__(self, name, args):
        super().__init__(args)
        self.gnn_name = name
        self.weight_decay = 0.

    def train(self, epoch):
        for model in self.models:
            model.train()
        self.optimizer.zero_grad()

        encoded_source = self.encode(self.source_data, "source")
        source_logits = self.models[1](encoded_source)

        # use source classifier loss:
        loss = self.loss_func(source_logits[self.label_mask], self.source_data.y[self.label_mask])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def init_model(self, label_rate=None):
        if label_rate is not None:
            print(f'init model with rate {label_rate}')
            self.label_rate = label_rate
        source_train_size = int(self.source_data.size(0) * self.label_rate)
        label_mask = np.array(
            [1] * source_train_size + [0] * (self.source_data.size(0) - source_train_size)).astype(
            bool)
        np.random.shuffle(label_mask)
        self.label_mask = torch.tensor(label_mask).to(self.device)
        self.models = construct_gnn(self.gnn_name, self.encoder_dim, self.num_features, self.num_classes,
                                    self.device)
        params = itertools.chain(*[model.parameters() for model in self.models])
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)

    def encode(self, data, cache_name, mask=None):
        encoded_output = self.models[0](data.x.float(), data.edge_index)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output
