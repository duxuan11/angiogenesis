import os
import argparse
import importlib.util

import torch

from isegm.utils.exp import init_experiment

def main(rank):
    args = parse_args()
    print("model_path", args.model_path)
    model_script = load_module(args.model_path)
    model_base_name = getattr(model_script, 'MODEL_NAME', None)

    args.local_rank = rank
    cfg = init_experiment(args, model_base_name)

    torch.backends.cudnn.benchmark = True

    model_script.main(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model script.')

    parser.add_argument('--exp-name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=2,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--gpus', type=str, default='0', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    return parser.parse_args()


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    rank = int(0)
    main(rank)
