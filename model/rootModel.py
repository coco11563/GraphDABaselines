import numpy
import torch
import random
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from model.utils.DomainData import construct_src, construct_dst

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

task = [
    {'source': 'acmv9',
     'target': 'citationv1'},
    {'source': 'acmv9',
     'target': 'dblpv7'},
    {'source': 'citationv1',
     'target': 'acmv9'},
    {'source': 'citationv1',
     'target': 'dblpv7'},
    {'source': 'dblpv7',
     'target': 'acmv9'},
    {'source': 'dblpv7',
     'target': 'citationv1'}
]

default_config = {
    'encoder_dim': 512,
    'learning_rate': 1e-3,
    'weight_decay': 1e-3
}

seed_option = [1, 2, 3, 4, 5]

label_rates = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]

default_rate = 0.05

cuda = [0, 1]

data_name = ['acmv9', 'dblpv7', 'citationv1']


class Trainer:
    def __init__(self, args):
        self.encoder_dim = 512
        self.learning_rate = 1e-3
        self.weight_decay = 1e-3
        self.epochs = 200
        data = construct_src(args)
        self.num_features = data.num_features
        self.num_classes = data.num_classes

        self.label_rate = args.label_rate
        print('init trainer with label rate : {}'.format(self.label_rate))
        print('init trainer on device:' + str(args.cuda))
        self.device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        self.source_data = data[0].to(self.device)
        self.target_data = construct_dst(args)[0].to(self.device)
        self.loss_func = None
        self.label_mask = None
        self.optimizer = None
        self.models = None
        self.model_name = f'{args.source}=>{args.target}'

    def encode(self, data, cache_name, mask=None):
        pass

    def test(self, data, cache_name, mask=None):
        for model in self.models:
            model.eval()
        encoded_output = self.encode(data, cache_name, mask)
        logits = self.models[1](encoded_output)
        preds = logits.argmax(dim=1)
        labels = data.y if mask is None else data.y[mask]
        corrects = preds.eq(labels)
        accuracy = corrects.float().mean()
        macro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
        micro_f1 = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
        return accuracy.cpu().numpy(), macro_f1, micro_f1

    def train(self, epoch):
        pass

    def eval(self, epoch):
        source_correct, _, _ = self.test(self.source_data, "source", self.source_data.test_mask)
        target_correct, macro_f1, micro_f1 = self.test(self.target_data, "target")
        return source_correct, target_correct, macro_f1, micro_f1

    def init_model(self, label_rate=None):
        pass

    def label_run(self, label_rate):
        print('init models')
        self.init_model(label_rate)
        return self.run()

    def labels_run(self):
        ps = []
        for rate in label_rates:
            p = self.label_run(rate)
            ps.append(p)
            print(f'done one rate : {rate}')
        name = ['best_target_acc', 'best_source_acc', 'best_micro_f1', 'best_macro_f1']
        for i, p in enumerate(ps):
            for j, n in enumerate(name):
                print(
                    f'{self.model_name}_{label_rates[i]} :'
                    f' {n} : {(p[j] * 100).round(2)}')
            print(f'best epoch of rate-{label_rates[i]} : {p[-1]}')
        return ps


    def run(self):
        best_macro_f1 = -0
        best_epoch = -1
        best_source_acc = -1
        best_target_acc = -1
        best_micro_f1 = -1
        for epoch in tqdm(range(1, self.epochs)):
            self.train(epoch)
            source_correct, target_correct, macro_f1, micro_f1 = self.eval(epoch)
            if macro_f1 > best_macro_f1:
                best_target_acc = target_correct
                best_source_acc = source_correct
                best_macro_f1 = macro_f1
                best_micro_f1 = micro_f1
                best_epoch = epoch
        return [best_target_acc,
                best_source_acc,
                best_micro_f1, best_macro_f1, best_epoch]

    def seeds_run(self):
        ps = [[], [], [], [], []]
        for seed in seed_option:
            p = self.seed_run(seed)
            for i, best in enumerate(p):
                ps[i].append(best)
            print('done one seed')
        name = ['best_target_acc', 'best_source_acc', 'best_micro_f1', 'best_macro_f1']
        for i, p in enumerate(ps[0:-1]):
            print(
                f'{self.model_name}-{self.label_rate} : {name[i]} : {(numpy.mean(p) * 100).round(2)}(+-{(numpy.std(p) * 100).round(2)}) max {(numpy.max(p) * 100).round(2)}')
        for i, s in enumerate(seed_option):
            print(f'best epoch of seed-{s} : {ps[-1][i]}')
        return ps

    def seed_run(self, seed):
        print('init seed with {}'.format(seed))
        self.init_seed(seed)
        print('init models')
        self.init_model()
        return self.run()

    def init_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

