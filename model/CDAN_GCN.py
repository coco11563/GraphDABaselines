import itertools
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn

from model.CDAN import CDANTrainer
from model.rootModel import task
from model.utils.AdvNet import AdversarialNetwork
from model.utils.Graph import GNN


def construct_cdan_gcn(encoder_dim, num_features, num_classes, device):
    encoder = GNN(encoder_dim, num_features, type="gcn").to(device)
    cls_model = nn.Sequential(
        nn.Linear(encoder_dim, num_classes),
    ).to(device)
    models = [encoder, cls_model]
    ad_net = AdversarialNetwork(encoder_dim * num_classes, encoder_dim).to(device)

    return models, ad_net

class CDANGCNTrainer(CDANTrainer):
    def __init__(self, args):
        super().__init__(args)

    def encode(self, data, cache_name, mask=None):
        encoded_output = self.models[0](data.x, data.edge_index, cache_name)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output

    def init_model(self, label_rate=None):
        if label_rate is not None:
            print(f'init model with rate {label_rate}')
            self.label_rate = label_rate
        source_train_size = int(self.source_data.size(0) * self.label_rate)
        label_mask = np.array([1] * source_train_size + [0] * (self.source_data.size(0) - source_train_size)).astype(
            bool)
        np.random.shuffle(label_mask)
        self.label_mask = torch.tensor(label_mask).to(self.device)
        self.models, self.ad_net = construct_cdan_gcn(self.encoder_dim, self.num_features, self.num_classes, self.device)
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

        trainer = CDANGCNTrainer(parser.parse_args())
        trainer.labels_run()

