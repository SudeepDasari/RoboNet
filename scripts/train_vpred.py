from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from robonet.video_prediction.training import GIFLogger, get_trainable
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
    parser.add_argument('--train_class', type=str, default='VPredTrainable', help='trainable type (specify for customizable train loop behavior)')
    parser.add_argument("--input_dir", type=str, default='./', help="directory containing video prediction data")
    parser.add_argument('--experiment_dir', type=str, default='./', help='directory containing model and dataset hparams')
    parser.add_argument('--save_freq', type=int, default=10000, help="how frequently to save model weights")
    parser.add_argument('--image_summary_freq', type=int, default=1000, help="how frequently to save image summaries")
    parser.add_argument('--scalar_summary_freq', type=int, default=100, help="how frequently to take validation summaries")

    parser.add_argument('--local_mode', action='store_true', help='runs ray in local mode if flag is supplied')
    parser.add_argument('--cluster', action='store_true', help='runs ray in cluster mode (by supplying redis_address) if flag is supplied')
    parser.add_argument('--no_resume', action='store_true', help='prevents ray from resuming (or restarting trials which crashed)')

    parser.add_argument('--batch_size', type=int, nargs='+', default=[8], help='batch size for model training (if list will grid search)')
    parser.add_argument('--max_steps', type=int, nargs='+', default=[300000], help="maximum number of iterations to train for (if list will grid search)")
    parser.add_argument('--train_frac', type=float, nargs='+', default=[0.9], help='fraction of data to use as training set (if list will grid search)')
    parser.add_argument('--val_frac', type=float, nargs='+', default=[0.05], help='fraction of data to use as validation set (if list will grid search)')

    parser.add_argument('--robot', type=str, default='', help="robot data to train on (if only one robot is desired)")
    parser.add_argument('--action_primitive', type=str, default='', help="if flag is supplied only trajectories with metadata['primitive']==action_primitive will be considered")
    parser.add_argument('--filter_adim', type=int, default=0, help="if flag is supplied only trajectories with adim=filter_adim will be trained on")
    args = parser.parse_args()

    dataset_hparams = json_try_load(args.experiment_dir + '/dataset_hparams.json')
    model_hparams = json_try_load(args.experiment_dir + '/model_hparams.json')
    
    config = {'dataset_hparams': dataset_hparams,
              'model_hparams': model_hparams,
              'train_fraction': tune.grid_search(args.train_frac),
              'batch_size': tune.grid_search(args.batch_size),
              'val_fraction': tune.grid_search(args.val_frac),
              'max_steps': tune.grid_search(args.max_steps),
              'data_directory': args.input_dir,
              'image_summary_freq': args.image_summary_freq,
              'scalar_summary_freq': args.scalar_summary_freq,
              'robot': args.robot,
              'action_primitive': args.action_primitive,
              'filter_adim': args.filter_adim}
    
    exp = tune.Experiment(
                name="{}_video_prediction_training".format(os.getlogin()),
                run=get_trainable(args.train_class),
                trial_name_creator=tune.function(trial_str_creator),
                loggers=[GIFLogger],
                config=config,
                resources_per_trial= {"cpu": 1, "gpu": 1},
                checkpoint_freq=args.save_freq)
    
    redis_address = None
    if args.cluster:
        redis_address = ray.services.get_node_ip_address() + ':6379'
    ray.init(redis_address=redis_address, local_mode=args.local_mode)

    trials = tune.run(exp, queue_trials=True, resume=not args.no_resume, checkpoint_at_end=True)
    exit(0)
