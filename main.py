import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.model import Model
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

# tf.enable_eager_execution()


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config("C:\\Pets\\configs\\config.json")

    except Exception as e:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.export_dir])
    # create your data generator
    # dataset = DataGenerator(config)
    # data = dataset.dataset.make_initializable_iterator().get_next()

    # create an instance of the model you want
    model = Model(config)
    if(config.mode == "train"):
        model.training(config)
    if(config.mode == "validation"):
        model.validate(config)
    if(config.mode == "test"):
        model.test(config)
    if(config.mode == "train_and_eval"):
        model.train_and_eval(config)

    # # create tensorflow session
    # sess = tf.Session()
    # # create tensorboard logger
    # logger = Logger(sess, config)
    # # create trainer and pass all the previous components to it
    # trainer = ExampleTrainer(sess, model, data, config, logger)
    # # load model if exists
    # model.load(sess)
    # # here you train your model
    # trainer.train()


if __name__ == '__main__':
    main()
