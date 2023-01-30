from argparse import ArgumentParser

from model.DANN_GCN import DANNGCNTrainer
from model.GNN import GNNTrainer
from model.rootModel import task

for t in task:
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default=t['source'])
    parser.add_argument("--target", type=str, default=t['target'])
    parser.add_argument("--label_rate", type=float, default=0.05)
    parser.add_argument("--cuda", type=int, default=0)

    trainer = GNNTrainer('gat' ,parser.parse_args())
    trainer.seeds_run()
