import numpy as np
import os

#from tflite_model_maker.config import ExportFormat
#from tflite_model_maker import model_spec
#from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

from termcolor import colored
from cookit.data import get_random_slice


class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.model = None
        self.spec = model_spec.get('efficientdet_lite4')

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pass

    def run(self):
        """fits model"""
        pass

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test"""

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        # see https://www.tensorflow.org/lite/tutorials/model_maker_object_detection#export_to_different_formats
        # model.export(export_dir='.', export_format=[ExportFormat.SAVED_MODEL, ExportFormat.LABEL])
        print(colored("model.joblib saved locally", "green"))


if __name__ == "__main__":
    # Get and clean data
    df = get_random_slice(nrows=1000)

    y = df["classes"]
    X = df.drop("classes", axis=1)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train and save model, locally and
    trainer = Trainer(X_train, y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    score = trainer.evaluate(X_test, y_test)
    print(f"Score of model : {score}")

    trainer.save_model_locally()
