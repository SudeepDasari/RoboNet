from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from robonet.video_prediction.training import GIFLogger, VPredTrainable
import json
import os
import tensorflow as tf
import ray
import ray.tune as tune


def json_try_load(fname):
    try:
        return json.load(open(fname, 'r'))
    except FileNotFoundError:
        return {}


def trial_str_creator(trial):
    return "{}_{}".format(str(trial), trial.trial_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="directory containing video prediction data")
    parser.add_argument('--save_freq', type=int, default=10000, help="how frequently to save model weights")
    parser.add_argument('--experiment_dir', type=str, default='', help='directory containing hparams')    # TODO add seperate model/dataset hparams loading
    parser.add_argument('--log_dir', type=str, default='', help="where to save model (will save in experiment_dir otherwise)")
    args = parser.parse_args()
    
    dataset_hparams = json_try_load(args.experiment_dir + '/dataset_hparams.json')
    model_hparams = json_try_load(args.experiment_dir + '/model_hparams.json')
    
    config = {'dataset_hparams': dataset_hparams,
              'model_hparams': model_hparams,
              'n_gpus': 1,
              'train_frac': tune.grid_search([0.9, 0.05]),
              'data_directory': args.input_dir,
              'batch_size': 16}

    
    exp = tune.Experiment(
                name="{}_video_prediction_training".format(os.getlogin()),
                run=VPredTrainable,
                trial_name_creator=tune.function(trial_str_creator),
                loggers=[GIFLogger],
                config=config,
                resources_per_trial= {"cpu": 10, "gpu": 1},
                checkpoint_freq=args.save_freq)
    
    redis_address = ray.services.get_node_ip_address() + ':6379'
    print('init ray on {}'.format(redis_address))
    ray.init(redis_address=redis_address, local_mode=True)

    trials = tune.run(exp, queue_trials=True)