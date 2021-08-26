import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
from cookit.data import get_data
from cookit.utils import OIv4_FOOD_CLASSES

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Predictor():
    def __init__(self):
        """
        A basic call for predictions.
        """
        self.model = self._get_model()

    def _get_model(self, path_to_joblib='model.joblib'):
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        return hub.load(module_handle).signatures['default']

    def load_img(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def resize_images(self, path, new_width=256, new_height=256):
        pil_image = Image.open(path)
        if pil_image.width > new_width or pil_image.height > new_height:
            pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
            pil_image_rgb = pil_image.convert("RGB")
            pil_image_rgb.save(path, format="JPEG", quality=90)
        return path

    def predict(self, filename, threshold=0.25):
        filename = self.resize_images(filename)
        print(f"Received file for prediction: {filename}")
        classes, scores = self.run_detector(filename)
        ingredients_raw = [i.decode('UTF-8') for i in classes]

        # sort out redundant labels but keep order
        ingredients = []
        for i, label in enumerate(ingredients_raw):
            # add only ingredients which have higher score than threshold
            if scores[i] > threshold:
                # add only ingredients which are food-related
                if label not in ingredients and label in OIv4_FOOD_CLASSES:
                    ingredients.append(label)
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
