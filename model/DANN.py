import itertools

import numpy as np
import torch
import torch.nn as nn

from model.rootModel import Trainer

rate = None

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


def construct_dann(encoder_dim, num_features, num_classes, device):
    encoder = nn.Sequential(
        nn.Linear(num_features, encoder_dim),
        nn.Linear(encoder_dim, encoder_dim)
    ).to(device)

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

    models = [encoder, cls_model, domain_model]
    return models


class DANNTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

    def train(self, epoch):
        for model in self.models:
            model.train()
        self.optimizer.zero_grad()
        global rate
        rate = min((epoch + 1) / self.epochs, 0.05)
        encoded_source = self.encode(self.source_data, "source")
        encoded_target = self.encode(self.target_data, "target")
        source_logits = self.models[1](encoded_source)

        # use source classifier loss:
        cls_loss = self.loss_func(source_logits[self.label_mask], self.source_data.y[self.label_mask])

        # use domain classifier loss:
        source_domain_preds = self.models[2](encoded_source)
        target_domain_preds = self.models[2](encoded_target)

        source_domain_cls_loss = self.loss_func(
            source_domain_preds,
            torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
        )
        target_domain_cls_loss = self.loss_func(
            target_domain_preds,
            torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
        )
        loss_grl = source_domain_cls_loss + target_domain_cls_loss
        loss = cls_loss + loss_grl

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def init_model(self, label_rate=None):
        if label_rate is not None:
            print(f'init model with rate {label_rate}')
            self.label_rate = label_rate
        source_train_size = int(self.source_data.size(0) * self.label_rate)
        label_mask = np.array([1] * source_train_size + [0] * (self.source_data.size(0) - source_train_size)).astype(
            bool)
        np.random.shuffle(label_mask)
        self.label_mask = torch.tensor(label_mask).to(self.device)
        self.models = construct_dann(self.encoder_dim, self.num_features, self.num_classes,
                                     self.device)
        params = itertools.chain(*[model.parameters() for model in self.models])
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)

    def encode(self, data, cache_name, mask=None):
        encoded_output = self.models[0](data.x)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output
