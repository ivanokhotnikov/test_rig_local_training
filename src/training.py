import argparse

from dotenv import load_dotenv

from run import Run


def parse_arguments():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument('--seq-len', type=int, default=120)
    parser.add_argument('--pred-len', type=int, default=15)
    parser.add_argument('--features', choices=['M', 'S', 'MS'], default='S')
    parser.add_argument('--target', type=str, default='LOAD_POWER')
    # architecture
    parser.add_argument('--hidden-size', type=int, default=50)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=.2)
    # training
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--exps', type=int, default=1)
    parser.add_argument('--val-split', type=float, default=.2)
    parser.add_argument('--test-split', type=float, default=.2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=.9)
    # ignore ipykernel
    parser.add_argument('--ip', default=argparse.SUPPRESS)
    parser.add_argument('--stdin', default=argparse.SUPPRESS)
    parser.add_argument('--control', default=argparse.SUPPRESS)
    parser.add_argument('--hb', default=argparse.SUPPRESS)
    parser.add_argument('--Session.signature_scheme',
                        default=argparse.SUPPRESS)
    parser.add_argument('--Session.key', default=argparse.SUPPRESS)
    parser.add_argument('--shell', default=argparse.SUPPRESS)
    parser.add_argument('--transport', default=argparse.SUPPRESS)
    parser.add_argument('--iopub', default=argparse.SUPPRESS)
    parser.add_argument('--f', default=argparse.SUPPRESS)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    load_dotenv()
    for ii in range(args.exps):
        args.seed += ii
        run = Run(args)
        run.train()
        # run.test()
        if args.dry_run: break
