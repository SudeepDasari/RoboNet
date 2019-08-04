import argparse
from robonet.video_prediction.training import GIFLogger, get_trainable
import tensorflow as tf
import ray
import ray.tune as tune
from yaml_util import parse_tune_config as parse_config
import os


def trial_str_creator(trial):
    return "{}_{}".format(str(trial), trial.trial_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_file', type=str, help='path to YAML experiment config file')
    parser.add_argument('--local_mode', action='store_true', help="if flag enables local_mode")
    parser.add_argument('--cluster', action='store_true', help="if flag enables cluster mode")
    parser.add_argument('--temp_dir', type=str, default=None, help="sets temp dir for ray redis (useful if permission error in /tmp/)")
    parser.add_argument('--name', type=str, default=None, help="sets experiment name")
    args = parser.parse_args()
    config = parse_config(args.experiment_file)

    redis_address, max_failures, local_mode = None, 10, False
    resume = config.pop('resume', False)
    if args.cluster or config.pop('cluster', False):
        redis_address = ray.services.get_node_ip_address() + ':6379'
        max_failures, resume = 20, True
    elif args.local_mode or config.pop('local_mode', False):
        local_mode = True
        max_failures = 0
    
    if args.temp_dir is None:
        args.temp_dir = config.pop('temp_dir', None)

    if args.name is not None:
        name = args.name
        config.pop('name', None)
    else:
        name = config.pop('name', "{}_video_prediction_training".format(os.getlogin()))

    exp = tune.Experiment(
                name=name,
                run=get_trainable(config.pop('train_class')),
                trial_name_creator=tune.function(trial_str_creator),
                loggers=[GIFLogger],
                resources_per_trial= {"cpu": 1, "gpu": 1},
                checkpoint_freq=config.pop('save_freq', 5000),
                upload_dir=config.pop('upload_dir', None),
                local_dir=config.pop('local_dir', None),
                config=config                                   # evaluate last to allow all popping above
    )
    
    ray.init(redis_address=redis_address, local_mode=local_mode, temp_dir=args.temp_dir)
    trials = tune.run(exp, queue_trials=True, resume=resume,
                      checkpoint_at_end=True, max_failures=max_failures)
    exit(0)
