#%%
import argparse

parser = argparse.ArgumentParser()
# task
parser.add_argument('--seq-len', type=int, default=300)
parser.add_argument('--pred-len', type=int, default=120)
parser.add_argument('--features', choices=['M', 'S', 'MS'], default='S')
parser.add_argument('--target', type=str, default='LOAD_POWER')
# architecture
parser.add_argument('--hidden-size', type=int, default=5)
parser.add_argument('--num-layers', type=int, default=1)
# training
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dry-run', action='store_true', default=False)
parser.add_argument('--exps', type=int, default=1)
parser.add_argument('--val-split', type=float, default=.3)
parser.add_argument('--test-split', type=float, default=.3)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--log-interval', type=int, default=500)
parser.add_argument('--learning-rate', type=float, default=1e-1)
# ignore ipykernel
parser.add_argument('--ip', default=argparse.SUPPRESS)
parser.add_argument('--stdin', default=argparse.SUPPRESS)
parser.add_argument('--control', default=argparse.SUPPRESS)
parser.add_argument('--hb', default=argparse.SUPPRESS)
parser.add_argument('--Session.signature_scheme', default=argparse.SUPPRESS)
parser.add_argument('--Session.key', default=argparse.SUPPRESS)
parser.add_argument('--shell', default=argparse.SUPPRESS)
parser.add_argument('--transport', default=argparse.SUPPRESS)
parser.add_argument('--iopub', default=argparse.SUPPRESS)
parser.add_argument('--f', default=argparse.SUPPRESS)

args = parser.parse_args()

#%%
import os

from exp import Experiment

os.chdir(os.path.dirname(os.getcwd()))

for ii in range(args.exps):
    setting = 'ft{}_t{}_sl{}_pl{}_{}'.format(args.features, args.target,
                                             args.seq_len, args.pred_len,
                                             ii + 1)
    print('\nexperiment: {}'.format(ii + 1))
    exp = Experiment(args)
    print('start training')
    exp.train(setting)
    print('start testing')
    exp.test()
    if args.dry_run: break
