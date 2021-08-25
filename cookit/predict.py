# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

import joblib
from cookit.data import get_data


class Predictor():
    def __init__(self):
        """
        A basic call for predictions.
        """
        self.model = self._get_model()

    def _get_model(self, path_to_joblib='model.joblib'):
        #pipeline = joblib.load(path_to_joblib)
        #return pipeline
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        return hub.load(module_handle).signatures['default']

    def load_img(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def predict(self, filename):
        print(f"Received file for prediction: {filename}")
        classes, scores = self.run_detector(filename)
        ingredients_raw = [i.decode('UTF-8') for i in classes]

        # sort out redundant labels but keep order
        ingredients = []
        for i in ingredients_raw:
            if i not in ingredients:
                ingredients.append(i)
        return ingredients
        #return ['Cucumber, Carrot, Garlic, Butter, Toast']

    def run_detector(self, path):
        img = self.load_img(path)

        converted_img = tf.image.convert_image_dtype(img,
                                                     tf.float32)[tf.newaxis,
                                                                 ...]
        result = self.model(converted_img)

        result = {key: value.numpy() for key, value in result.items()}
        return (result["detection_class_entities"], result["detection_scores"])


if __name__ == '__main__':
    df_test = get_data('path_to_test_data')
    predictor = Predictor()
    y_pred = predictor.predict(df_test)

    print(f"Prediction: {y_pred}")
