import tensorflow as tf
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from termcolor import colored
from cookit.data import get_random_slice
from cookit.params import BUCKET_NAME

class Trainer(object):
    def __init__(self):
        """

        """
        self.model = None
        self._spec = model_spec.get('efficientdet_lite4')

    def load_data(self, csv_path=f'gs://{BUCKET_NAME}/oi_food_converted_sample.csv'):
        data = object_detector.DataLoader.from_csv(csv_path)
        self.train_data = data[0]
        self.val_data = data[1]
        self.test_data = data[2]

    def run(self, batch_size=32, train_whole_model=True):
        """fits model"""
        self.model = object_detector.create(self.train_data,
                                       model_spec=self._spec,
                                       batch_size=batch_size,
                                       train_whole_model=train_whole_model,
                                       validation_data=self.val_data)

    def evaluate(self):
        """evaluates the pipeline on df_test"""
        self.model.evaluate(self.test_data)

    def save_model_locally(self, model_name='model.tflite'):
        """Save the model into a .joblib format"""
        # see https://www.tensorflow.org/lite/tutorials/model_maker_object_detection#export_to_different_formats
        # model.export(export_dir='.', export_format=[ExportFormat.SAVED_MODEL, ExportFormat.LABEL])
        self.model.export(export_dir='.',
                          tflite_filename='model.tflite',
                          label_filename='labels.txt',
                          saved_model_filename=model_name,
                          #export_format=None,
                          )

        print(colored(f"Saved trained model to {model_name}", "green"))


if __name__ == "__main__":
    pass
    # Get and clean data
    # df = get_random_slice('infile', 'outfile', size=1000)


    # # Train and save model, locally and
    # trainer = Trainer()
    # trainer.run()
    # score = trainer.evaluate()
    # print(f"Score of model : {score}")
    # trainer.save_model_locally()
