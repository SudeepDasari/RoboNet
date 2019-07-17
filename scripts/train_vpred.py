import argparse
from robonet.video_prediction.training import GIFLogger, get_trainable
import tensorflow as tf
import ray
import ray.tune as tune
import re, yaml, os, json


def parse_config(config_file):
    """
    Configures custom yaml loading behavior and parses config file
    """
    search_pattern = re.compile(r".*search\/(.*?)\((.*?)\)", re.VERBOSE)
    def search_constructor(loader, node):
        value = loader.construct_scalar(node)
        search_type, args = search_pattern.match(value).groups()
        if search_type == 'grid':
            return tune.grid_search(json.loads(args))
        raise NotImplementedError("search {} is not implemented".format(search_type))
    yaml.add_implicit_resolver("!custom_search", search_pattern, Loader=yaml.SafeLoader)
    yaml.add_constructor('!custom_search', search_constructor, Loader=yaml.SafeLoader)

    env_pattern = re.compile(r"\$\{(.*?)\}(.*)", re.VERBOSE)
    def env_var_constructor(loader, node):
        """
        Converts ${VAR}/* from config file to 'os.environ[VAR] + *'
        Modified from: https://www.programcreek.com/python/example/61563/yaml.add_implicit_resolver
        """
        value = loader.construct_scalar(node)
        env_var, remainder = env_pattern.match(value).groups()
        if env_var not in os.environ:
            raise ValueError("config requires envirnonment variable {} which is not set".format(env_var))
        return os.environ[env_var] + remainder
    yaml.add_implicit_resolver("!env", env_pattern, Loader=yaml.SafeLoader)
    yaml.add_constructor('!env', env_var_constructor, Loader=yaml.SafeLoader)

    with open(config_file) as config:
        return yaml.load(config, Loader=yaml.SafeLoader) 

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

    redis_address, max_failures, local_mode = None, 3, False
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
