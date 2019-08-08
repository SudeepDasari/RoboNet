import functools
import os
from robonet.datasets import get_dataset_class
from robonet.video_prediction.models import get_model
import tensorflow as tf
from yaml_util import parse_tpu_config as parse_config


def dataset_fn(params, DatasetClass, batch_sizes, loader_files, dataset_hparams):
    loader = DatasetClass(batch_sizes, loader_files, dataset_hparams)
    inputs = {}
    targets = {}

    inputs['actions'] = loader['actions']
    inputs['images'] = loader['images'][:, :-1]
    inputs['states'] = loader['states'][:, :-1]

    targets['images'] = loader['images']
    targets['states'] = loader['states']

    return inputs, targets


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='launches video prediction training on tpu instances')
    parser.add_argument('experiment_file', type=str, default='',  help='path of experiment file')
    parser.add_argument('--testing', action='store_true', help='if flag is supplied then assume testing mode (model run on cpu)')
    args = parser.parse_args()
    
    config = parse_config(args.experiment_file)
    dataset_hparams = config.pop('loader_hparams')
    model_hparams = config.pop('model_hparams')

    # add bucket_dir to hparams
    if 'BUCKET' in os.environ and 'bucket_dir' not in dataset_hparams:
        dataset_hparams['bucket_dir'] = os.environ['BUCKET']
        config['save_dir'] = '{}/{}'.format(os.environ['BUCKET'], config['save_dir'])

    # extract train params from config
    input_dir = os.path.expanduser(config['data_directory'])
    batch_sizes = config['batch_sizes']
    model_hparams['summary_dir'] = save_dir = os.path.expanduser(config['save_dir'])
    iter_per_loop = config.get('iter_per_loop', 100)
    train_steps_per_eval = config.get('train_steps_per_eval', 1000)
    robots = config.get('robots', ['sawyer'])
    max_steps = config.get('max_steps', 300000)

    loader_files = ['{}/{}'.format(input_dir, r) for r in robots]
    DatasetClass = get_dataset_class(dataset_hparams.pop('dataset', 'TPU'))
    
    train_input = functools.partial(dataset_fn, DatasetClass=DatasetClass, batch_sizes=batch_sizes, 
                                    loader_files=loader_files, dataset_hparams=dataset_hparams)
    
    PredictionModel = get_model(model_hparams.pop('model'))
    model = PredictionModel(None, 0, model_hparams.pop('graph_type'), True, '')

    tpu_cluster_resolver=None
    if not args.testing:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(os.environ['TPU_NAME'], zone=os.environ['TPU_ZONE'], project=os.environ['PROJECT_ID'])

    tpu_config = tf.contrib.tpu.TPUConfig(iterations_per_loop=iter_per_loop)
    run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver, model_dir=save_dir, save_checkpoints_steps=train_steps_per_eval,tpu_config=tpu_config)

    tf.logging.set_verbosity(tf.logging.DEBUG)
    estimator = tf.contrib.tpu.TPUEstimator(model_fn=model.model_fn,
                                            use_tpu=not args.testing,
                                            train_batch_size=sum(batch_sizes),
                                            eval_batch_size=sum(batch_sizes),
                                            predict_batch_size=sum(batch_sizes),
                                            params=model_hparams,
                                            config=run_config)
    
    estimator.train(input_fn=train_input, max_steps=max_steps)