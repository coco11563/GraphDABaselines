import itertools
from argparse import ArgumentParser

import itertools
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rootModel import Trainer, task
from model.utils.AdvNet import AdversarialNetwork
# from model.utils.Graph import GRL


def construct_cdan(encoder_dim, num_features, num_classes, device):
    encoder_dim = encoder_dim

    encoder = nn.Sequential(
        nn.Linear(num_features, encoder_dim),
        nn.Linear(encoder_dim, encoder_dim)
    ).to(device)

    cls_model = nn.Sequential(
        nn.Linear(encoder_dim, num_classes),
    ).to(device)

    models = [encoder, cls_model]

    ad_net = AdversarialNetwork(encoder_dim * num_classes, encoder_dim).cuda()

    return models, ad_net


class CDANTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.ad_net = None

    def encode(self, data, cache_name, mask=None):
        encoded_output = self.models[0](data.x)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output

    def train(self, epoch):
        for model in self.models:
            model.train()
        self.optimizer.zero_grad()

        encoded_source = self.encode(self.source_data, "source")
        encoded_target = self.encode(self.target_data, "target")
        source_logits = self.models[1](encoded_source)
        target_logits = self.models[1](encoded_target)
        source_probs = F.softmax(source_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)

        # use source classifier loss:
        cls_loss = self.loss_func(source_logits[self.label_mask], self.source_data.y[self.label_mask])

        probs = torch.cat((source_probs, target_probs), dim=0)
        features = torch.cat((encoded_source, encoded_target), dim=0)

        op_out = torch.bmm(probs.unsqueeze(2), features.unsqueeze(1))
        ad_out = self.ad_net(op_out.view(-1, probs.size(1) * features.size(1)))

        dc_target = torch.from_numpy(np.array([[1]] * self.source_data.size(0) +
                                              [[0]] * self.target_data.size(0))).float().cuda()

        loss_grl = nn.BCELoss()(ad_out, dc_target)

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
        self.models, self.ad_net = construct_cdan(self.encoder_dim, self.num_features, self.num_classes, self.device)
        params = itertools.chain(*[model.parameters() for model in self.models])
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)

if __name__ == '__main__':
    for t in task:
        parser = ArgumentParser()
        parser.add_argument("--source", type=str, default=t['source'])
        parser.add_argument("--target", type=str, default=t['target'])
        parser.add_argument("--label_rate", type=float, default=0.05)
        parser.add_argument("--cuda", type=int, default=0)

        trainer = CDANTrainer(parser.parse_args())
        trainer.labels_run()