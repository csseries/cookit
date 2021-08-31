import os
import tensorflow as tf
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import pycocotools  # this is needed for model.evaluate, even if not explicitley used in code!
from termcolor import colored

from cookit.data import upload_file_to_bucket
from cookit.params import BUCKET_NAME

# make tensorflow less verbose
tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




class Trainer(object):
    def __init__(self, spec='efficientdet_lite0'):
        """

        """
        self.model = None
        self._spec = model_spec.get(spec)
        self._cache_prefix = 'cookit_trainer'

    def load_data(self, csv_path=f'gs://{BUCKET_NAME}/oi_food_converted_sample.csv',
                  force_download=False):
        cache_dir = csv_path.split('/')[-1].rstrip('.csv')
        if not os.path.isdir(cache_dir) and force_download == False:
            print(f"Downloading images from {csv_path}")

            data = object_detector.DataLoader.from_csv(csv_path,
                                                    cache_dir=cache_dir,
                                                    cache_prefix_filename=self._cache_prefix)
            self.train_data = data[0]
            self.val_data = data[1]
            self.test_data = data[2]
            self.label_map = self.train_data.label_map
        else:
            self.load_data_from_cache(cache_dir)

    def load_data_from_cache(self, cache_dir):
        print(f"Load images from {cache_dir}")
        cache_prefix = f"{cache_dir}/train_{self._cache_prefix}"
        self.train_data = object_detector.DataLoader.from_cache(cache_prefix)
        cache_prefix = f"{cache_dir}/test_{self._cache_prefix}"
        self.test_data = object_detector.DataLoader.from_cache(cache_prefix)
        cache_prefix = f"{cache_dir}/val_{self._cache_prefix}"
        self.val_data = object_detector.DataLoader.from_cache(cache_prefix)
        self.label_map = self.train_data.label_map

    def run(self, epochs=50, batch_size=32, train_whole_model=True):
        """fits model"""
        self.model = object_detector.create(self.train_data,
                                            epochs=epochs,
                                            model_spec=self._spec,
                                            batch_size=batch_size,
                                            train_whole_model=train_whole_model,
                                            validation_data=self.val_data,
                                            label_map=self.label_map)

    def evaluate(self):
        """evaluates the pipeline on df_test"""
        eval_dict = self.model.evaluate(self.test_data)
        print(eval_dict)
        return eval_dict

    def save_model_locally(self, model_name='model.tflite', label_filename='labels.txt'):
        """Save the model into a .joblib format"""
        # see https://www.tensorflow.org/lite/tutorials/model_maker_object_detection#export_to_different_formats
        # model.export(export_dir='.', export_format=[ExportFormat.SAVED_MODEL, ExportFormat.LABEL])
        self.model.export(export_dir='.',
                          tflite_filename=model_name,
                          label_filename=label_filename,
                          #saved_model_filename=model_name,
                          #export_format=None,
                          )

        print(colored(f"Saved trained model to {model_name}", "green"))


if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_data('gs://taxifare_bucket_fast-drake-318911/oi_food_minimal_balanced.csv')
    trainer.run()
    eval_dict = trainer.evaluate()
    trainer.save_model_locally('model_min_8k_balanced.tflite')
    upload_file_to_bucket('model_min_8k_balanced.tflite')
    upload_file_to_bucket('labels.txt')
