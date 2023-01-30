import itertools

import numpy as np
import torch
import torch.nn as nn

from model.DANN import GRL
from model.DANN import DANNTrainer
from model.utils.Attention import Attention
from model.utils.Graph import GNN


def construct_udagcn(encoder_dim, num_features, num_classes, device):
    encoder = GNN(encoder_dim, num_features, type="gcn").to(device)
    ppmi_encoder = GNN(encoder_dim, num_features, base_model=encoder, type="ppmi", path_len=10).to(device)

    cls_model = nn.Sequential(
        nn.Linear(encoder_dim, num_classes),
    ).to(device)

    domain_model = nn.Sequential(
        GRL(),
        nn.Linear(encoder_dim, 40),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(40, 2),
    ).to(device)

    att_model = Attention(encoder_dim).cuda()
    models = [encoder, cls_model, domain_model, ppmi_encoder, att_model]
    return models

class UDAGCNTrainer(DANNTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.learning_rate=3e-3
        self.weight_decay=5e-4
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.params = None

    def encode(self, data, cache_name, mask=None):
        encoded_output = self.models[0](data.x, data.edge_index, cache_name)
        if mask is not None:
            encoded_output = encoded_output[mask]
        ppmi_output = self.models[3](data.x, data.edge_index, cache_name)
        if mask is not None:
            ppmi_output = ppmi_output[mask]
        outputs = self.models[4]([encoded_output, ppmi_output])
        return outputs

    def init_model(self, label_rate=None):
        if label_rate is not None:
            print(f'init model with rate {label_rate}')
            self.label_rate = label_rate
        source_train_size = int(self.source_data.size(0) * self.label_rate)
        label_mask = np.array([1] * source_train_size + [0] * (self.source_data.size(0) - source_train_size)).astype(
            bool)
        np.random.shuffle(label_mask)
        self.label_mask = torch.tensor(label_mask).to(self.device)
        self.models = construct_udagcn(self.encoder_dim, self.num_features, self.num_classes,
                                         self.device)
        params = itertools.chain(*[model.parameters() for model in self.models])
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)