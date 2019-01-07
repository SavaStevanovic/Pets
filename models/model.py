from base.base_model import BaseModel
import tensorflow as tf
import numpy as np
from data_loader.data_generator import DataGenerator
import pandas as pd


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.breed_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
                             164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307]

        self.build_model(config)

    def _get_feature_columns(self):
        types = tf.feature_column.categorical_column_with_vocabulary_list('Type', [1, 2])
        types = tf.feature_column.indicator_column(types)
        breed1 = tf.feature_column.categorical_column_with_vocabulary_list('Breed1', self.breed_labels)
        breed1 = tf.feature_column.embedding_column(breed1, dimension=5)
        breed2 = tf.feature_column.categorical_column_with_vocabulary_list('Breed2', self.breed_labels)
        breed2 = tf.feature_column.embedding_column(breed2, dimension=5)
        gender = tf.feature_column.categorical_column_with_vocabulary_list('Gender', [1, 2, 3])
        gender = tf.feature_column.indicator_column(gender)
        color1 = tf.feature_column.categorical_column_with_vocabulary_list('Color1', [1, 2, 3, 4, 5, 6, 7])
        color1 = tf.feature_column.indicator_column(color1)
        color2 = tf.feature_column.categorical_column_with_vocabulary_list('Color2', [1, 2, 3, 4, 5, 6, 7])
        color2 = tf.feature_column.indicator_column(color2)
        color3 = tf.feature_column.categorical_column_with_vocabulary_list('Color3', [1, 2, 3, 4, 5, 6, 7])
        color3 = tf.feature_column.indicator_column(color3)
        maturity_size = tf.feature_column.categorical_column_with_vocabulary_list('MaturitySize', [1, 2, 3, 4, 0])
        maturity_size = tf.feature_column.indicator_column(maturity_size)
        fur_length = tf.feature_column.categorical_column_with_vocabulary_list('FurLength', [1, 2, 3, 0])
        fur_length = tf.feature_column.indicator_column(fur_length)
        vaccinated = tf.feature_column.categorical_column_with_vocabulary_list('Vaccinated', [1, 2, 3])
        vaccinated = tf.feature_column.indicator_column(vaccinated)
        dewormed = tf.feature_column.categorical_column_with_vocabulary_list('Dewormed', [1, 2, 3])
        dewormed = tf.feature_column.indicator_column(dewormed)
        sterilized = tf.feature_column.categorical_column_with_vocabulary_list('Sterilized', [1, 2, 3])
        sterilized = tf.feature_column.indicator_column(sterilized)
        health = tf.feature_column.categorical_column_with_vocabulary_list('Health', [1, 2, 3, 0])
        health = tf.feature_column.indicator_column(health)
        state = tf.feature_column.categorical_column_with_vocabulary_list(
            'State', [41336, 41325, 41367, 41401, 41415, 41324, 41332, 41335, 41330, 41380, 41327, 41345, 41342, 41326, 41361])
        state = tf.feature_column.indicator_column(state)
        adoption_speed = tf.feature_column.categorical_column_with_vocabulary_list('AdoptionSpeed', [0, 1, 2, 3, 4])
        adoption_speed = tf.feature_column.indicator_column(adoption_speed)

        return [
            tf.feature_column.numeric_column('Age', default_value=-1),
            tf.feature_column.numeric_column('Quantity', default_value=-1),
            tf.feature_column.numeric_column('Fee', default_value=-1),
            tf.feature_column.numeric_column('VideoAmt', default_value=-1),
            tf.feature_column.numeric_column('PhotoAmt', default_value=-1),
            types,
            breed1,
            breed2,
            gender,
            color1,
            color2,
            color3,
            maturity_size,
            fur_length,
            vaccinated,
            dewormed,
            sterilized,
            health
        ]

    # def _set_input_fn(self, mode, config, data, labels=None):
    #     if(mode == 'train'):
    #         self.input_fn = tf.estimator.inputs.numpy_input_fn(
    #             x={"x": np.array(data)},
    #             y=np.array(labels),
    #             num_epochs=None,
    #             shuffle=True,
    #             batch_size=config.batch_size,
    #         )
    #     if(mode == 'validate'):
    #         self.input_fn = tf.estimator.inputs.numpy_input_fn(
    #             x={"x": np.array(data)},
    #             y=np.array(labels),
    #             num_epochs=1,
    #             shuffle=False,
    #         )
    #     if(mode == 'test'):
    #         self.input_fn = tf.estimator.inputs.numpy_input_fn(
    #             x={"x": np.array(data)},
    #             num_epochs=1,
    #             shuffle=False
    #         )

    def training(self, config):
        self.model.train(input_fn=DataGenerator(config.mode, config).get_dataset, steps=config.steps)

    def train_and_eval(self, config):

        # serving_feature_spec = tf.feature_column.make_parse_example_spec(self._get_feature_columns())
        # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec)

        # exporter = tf.estimator.BestExporter(
        #     name="best_exporter",
        #     serving_input_receiver_fn=serving_input_receiver_fn,
        #     exports_to_keep=5)
        tf.estimator.train_and_evaluate(
            self.model,
            train_spec=tf.estimator.TrainSpec(input_fn=DataGenerator('train', config).get_dataset),
            eval_spec=tf.estimator.EvalSpec(input_fn=DataGenerator(
                'validation', config).get_dataset, steps=1000, start_delay_secs=0, throttle_secs=1)
        )

    def validate(self, config):
        result = self.model.evaluate(DataGenerator(config.mode, config).get_dataset)
        print(result)

    def test(self, config):
        predictions = self.model.predict(DataGenerator(config.mode, config).get_dataset)
        cls = [p['classes'] for p in predictions]
        cls_pred = np.array(cls, dtype='int').squeeze()
        df = pd.read_csv(config.submission_file)
        df['AdoptionSpeed'] = cls_pred
        df.to_csv(config.submission_file)
        return cls_pred

    def build_model(self, config):
        self.model = tf.estimator.DNNClassifier(feature_columns=self._get_feature_columns(),
                                                hidden_units=config.hidden_units,
                                                activation_fn=tf.nn.relu,
                                                n_classes=5,
                                                optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
                                                dropout=config.dropout_probability,
                                                model_dir="./checkpoint/")