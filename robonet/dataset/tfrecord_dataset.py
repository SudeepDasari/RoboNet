import pickle as pkl
import tensorflow as tf
import os
import glob
from visual_mpc.datasets.base_dataset import BaseVideoDataset


def mult_elems(tup):
    prod = 1
    for t in tup:
        prod *= t
    return prod

class TFRecordDataset(BaseVideoDataset):
    def _init_dataset(self):
        # get and read manifest file
        self._read_manifest()

        # initialize batches (one tf Dataset per each mode where batches are drawn from)
        self._initialize_batches()

    def _get_default_hparams(self):
        default_params = super(TFRecordDataset, self)._get_default_hparams()
        
        default_params.add_hparam('compressed', True)
        default_params.add_hparam('sequence_length', None)
        
        return default_params

    def _parse_record(self, serialized_example):
        def get_feature(manifest_entry):
            shape, dtype = manifest_entry
            if dtype == 'Byte':
                return tf.FixedLenFeature([1], tf.string)
            elif dtype == 'Float':
                return tf.FixedLenFeature([mult_elems(shape)], tf.float32)
            elif dtype == 'Int':
                return tf.FixedLenFeature([mult_elems(shape)], tf.int64)
            raise ValueError('Unknown dtype: {}'.format(dtype))

        def decode_feat(feat, manifest_entry, pad_t=False):
            orig_shape, dtype = list(manifest_entry[0]), manifest_entry[1]
            shape = [s for s in orig_shape]
            if pad_t:
                shape = [1] + shape

            if dtype == 'Byte':
                uint_data = tf.decode_raw(feat, tf.uint8)
                img_flat = tf.reshape(uint_data, shape=[1, mult_elems(shape)])
                image = tf.reshape(img_flat, shape=orig_shape)
                image = tf.reshape(image, shape=shape)
                return image
            elif dtype == 'Float' or dtype == 'Int':
                return tf.reshape(feat, shape=shape)
            raise ValueError('Unknown dtype: {}'.format(dtype))

        features_names = {}
        for k in self._metadata_keys:
            features_names[k] = get_feature(self._metadata_keys[k])
        if self._T > 0:
            # print(self._T)
            for k in self._sequence_keys:
                for t in range(self._T):
                    features_names['{}/{}'.format(t, k)] = get_feature(self._sequence_keys[k])

        feature = tf.parse_single_example(serialized_example, features=features_names)

        return_dict = {}
        if self._T > 0:
            for k in self._sequence_keys:
                k_feats = []
                for t in range(self._T):
                    k_feat = decode_feat(feature['{}/{}'.format(t, k)], self._sequence_keys[k], True)
                    k_feats.append(k_feat)
                return_dict[k] = tf.concat(k_feats, 0)
        for k in self._metadata_keys:
            return_dict[k] = decode_feat(feature[k], self._metadata_keys[k])

        return return_dict

    def _initialize_batches(self):
        assert isinstance(self._files, str), "must pass a folder path into this reader!"
        
        self._raw_data = {}
        for m in self.MODES:
            fnames = glob.glob('{}/{}/*.tfrecords'.format(self._files, m))
            if len(fnames) == 0:
                print('Warning dataset does not have files for mode: {}'.format(m))
                continue

            if self._hparams.compressed:
                dataset = tf.data.TFRecordDataset(fnames, buffer_size=self._hparams.buffer_size, compression_type='GZIP')
            else:
                dataset = tf.data.TFRecordDataset(fnames, buffer_size=self._hparams.buffer_size)

            dataset = dataset.map(self._parse_record)
            dataset = dataset.repeat(self._hparams.num_epochs)
            dataset = dataset.shuffle(buffer_size=self._hparams.buffer_size)
            dataset = dataset.batch(self._batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            output_element = {}
            for k in list(next_element.keys()):
                output_element[k] = tf.reshape(next_element[k],
                                               [self._batch_size] + next_element[k].get_shape().as_list()[1:])

            self._raw_data[m] = output_element

    def _map_key(self, dataset_batch, key):
        if key == 'state' or key == 'endeffector_pos':
            return dataset_batch['env/state']
        elif key == 'actions':
            return dataset_batch['policy/actions']
        elif key == 'images':
            imgs, i = [], 0
            while True:
                image_name = 'env/image_view{}/encoded'.format(i)
                if image_name not in dataset_batch:
                    break
                imgs.append(tf.expand_dims(dataset_batch[image_name], 2))
                i += 1
            if i == 0:
                raise ValueError("No image tensors")
            elif i == 1:
                return imgs[0]
            return tf.concat(imgs, 2)

        elif key in dataset_batch:
            return dataset_batch[key]

        raise NotImplementedError('Key {} not present in batch which has keys:\n {}'.format(key,
                                                                                            list(dataset_batch.keys())))

    def _get(self, key, mode):
        dataset_batch = self._raw_data[mode]
        return self._map_key(dataset_batch, key)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise KeyError('Index should be in format: [Key, Mode] or [Key] (assumes default train mode)')
            key, mode = item
            return self.get(key, mode)

        return self.get(item)

    def get_iterator(self, item, mode):
        fnames = glob.glob('{}/{}/*.tfrecords'.format(self._base_dir, mode))
        if self._hparams.compressed:
            dataset = tf.data.TFRecordDataset(fnames, buffer_size=self._hparams.buffer_size,
                                              compression_type='GZIP')
        else:
            dataset = tf.data.TFRecordDataset(fnames, buffer_size=self._hparams.buffer_size)

        def parse_record(ex):
            return self._parse_record(ex)[item]


        dataset = dataset.map(parse_record)
        dataset = dataset.repeat(self._hparams.num_epochs)
        dataset = dataset.shuffle(buffer_size=self._hparams.buffer_size)
        dataset = dataset.batch(self._batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def _read_manifest(self):
        pkl_path = '{}/manifest.pkl'.format(self._base_dir)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError('Manifest not found at {}/manifest.pkl'.format(self._base_dir))

        manifest_dict = pkl.load(open(pkl_path, 'rb'))
        self._sequence_keys = manifest_dict['sequence_data']
        self._metadata_keys = manifest_dict['traj_metadata']
        if self._hparams.sequence_length is None:
            self._T = manifest_dict['T']
        else:
            self._T = self._hparams.sequence_length

    @property
    def T(self):
        return self._T


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='folder containing hdf5 files')
    args = parser.parse_args()

    path = args.input_folder
    batch_size = 1
    dataset = TFRecordDataset(path, batch_size)
    images, actions = dataset['images'], dataset['actions']
    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    for _ in range(10):
        i, a = sess.run([images, actions])
        print('batched actions have size', a.shape)