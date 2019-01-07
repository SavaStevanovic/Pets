import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder


class DataGenerator:
    def __init__(self, mode, config):
        self.config = config
        self.repeat = -1
        self.mode=mode
        if(mode == "test"):
            self.filenames = [config.test_data_path]
            self.record_defaults = [[-1]]+[['']] + [[-1]]*16+[['']]+[[-1]]+[['']]*2+[[-1.]]
            self.repeat = 1
        if(mode == "validation"):
            self.filenames = [config.validation_data_path]
            self.record_defaults = [[-1]]+[['']] + [[-1]]*16+[['']]+[[-1]]+[['']]*2+[[-1.]]+[[-1]]
            self.repeat = 1
        if(mode == "train"):
            self.filenames = [config.train_data_path]
            self.record_defaults = [[-1]]+[['']] + [[-1]]*16+[['']]+[[-1]]+[['']]*2+[[-1.]]+[[-1]]
        self.batch_size = config.batch_size
        self.workers = config.workers

    def _parse_function(self, *example_proto):
        features = dict(zip(self.col_names, example_proto))
        selected_features = {key: features[key] for key in self.feature_keys}
        

        # data = tf.feature_column.input_layer(features, columns)
        if(self.mode=='test'):
            return selected_features
        labels = features['AdoptionSpeed']
        return selected_features, labels

    # def next_batch(self, batch_size):
    #     idx = np.random.choice(500, batch_size)
    #     yield self.input[idx], self.dataset["AdoptionSpeed"]

    col_names = ['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                 'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'VideoAmt', 'Description', 'PetID', 'PhotoAmt', 'AdoptionSpeed']
    feature_keys = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                    'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt']

    def get_dataset(self):
        self.dataset = tf.contrib.data.CsvDataset(self.filenames, self.record_defaults, header=True)
        if(self.mode=='test'):
            return self.dataset.map(self._parse_function).batch(256)
        return self.dataset.apply(tf.contrib.data.map_and_batch(self._parse_function, self.batch_size, num_parallel_calls=12)).apply(tf.contrib.data.shuffle_and_repeat(4, self.repeat)).prefetch(1)
