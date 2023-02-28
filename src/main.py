import argparse

from exp import Experiment

parser = argparse.ArgumentParser()
# GPU
parser.add_argument('--use-gpu', action='store_true', default=None)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--use-multi-gpu', action='store_true', default=None)
# Task
parser.add_argument('--seq-len',
                    type=int,
                    default=120,
                    help='input sequence length')
parser.add_argument('--pred-len',
                    type=int,
                    default=1,
                    help='prediction sequence length')
parser.add_argument('--features',
                    choices=['M', 'S', 'MS'],
                    default='S',
                    help='forecasting task, options: [M, S, MS]')
parser.add_argument('--target',
                    type=str,
                    default=None,
                    help='forecasting target for features [S, MS]')
# Architecture
parser.add_argument(
    '--hidden-size',
    type=int,
    default=20,
    help='The number of features in the hidden state h of LSTM')
parser.add_argument('--num-layers',
                    type=int,
                    default=2,
                    help='Number of recurrent layers in LSTM')
# Training
parser.add_argument('--dry-run', action='store_true', default=False)
parser.add_argument('--itr', type=int, default=2)
parser.add_argument('--val-split', type=float, default=.2)
parser.add_argument('--test-split', type=float, default=.2)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument('--learning-rate', type=float, default=0.01)
# Ignore Ipykernel
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

if __name__ == '__main__':
    args = parser.parse_args()
    for k, v in vars(args).items():
        if all((k, v)): print('{}: {}'.format(k, v))
    for ii in range(args.itr):
        exp = Experiment(args)
        exp.train()
        if args.dry_run: break
