import itertools

import numpy as np
import torch
import torch.nn as nn

from model.rootModel import Trainer


def construct_mlp(encoder_dim, num_features, num_classes, device):
    encoder = nn.Sequential(
        nn.Linear(num_features, encoder_dim),
        nn.Linear(encoder_dim, encoder_dim)
    ).to(device)
    cls_model = nn.Sequential(
        nn.Linear(encoder_dim, num_classes),
    ).to(device)

    models = [encoder, cls_model]
    return models


class MLPTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.weight_decay=1e-4

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
        self.models = construct_mlp(self.encoder_dim, self.num_features, self.num_classes,
                                    self.device)
        params = itertools.chain(*[model.parameters() for model in self.models])
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)

    def train(self, epoch):
        for model in self.models:
            model.train()
        self.optimizer.zero_grad()
        encoded_source = self.encode(self.source_data, "source")
        source_logits = self.models[1](encoded_source)
        loss = self.loss_func(source_logits[self.label_mask], self.source_data.y[self.label_mask])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def encode(self, data, cache_name, mask=None):
        encoded_output = self.models[0](data.x)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output
