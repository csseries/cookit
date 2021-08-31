import tensorflow as tf
import numpy as np
import os

from cookit.utils import OIv4_FOOD_CLASSES

# Make tensorflow less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Predictor():
    def __init__(self):
        """
        A basic call for predictions.
        """
        self.model = self._get_model('model_min_8k_balanced.tflite')

        # NOTE: The order of this list hardcoded here, and needs to be changed when re-training the model!
        # When exporting the model in tflite format, the model_spec is lost, so we cannot do it like that:
        # classes = ['???'] * model.model_spec.config.num_classes
        # label_map = model.model_spec.config.label_map
        # for label_id, label_name in label_map.as_dict().items():
        #   classes[label_id-1] = label_name
        #self.classes = ['Baked Goods', 'Salad', 'Cheese', 'Seafood', 'Tomato']
        self.classes = [
            'Pumpkin', 'Ice cream', 'Salad', 'Bread', 'Coconut', 'Grape',
            'Mushroom', 'Honeycomb', 'Fish', 'Oyster', 'Pomegranate', 'Radish',
            'Watermelon', 'Pasta', 'Cabbage', 'Strawberry', 'Apple', 'Orange',
            'Potato', 'Banana', 'Pear', 'Shellfish', 'Tomato', 'Cheese',
            'Carrot', 'Shrimp', 'Lemon', 'Artichoke', 'Broccoli',
            'Bell pepper', 'Pineapple', 'Lobster', 'Milk', 'Mango',
            'Grapefruit', 'Cantaloupe', 'Peach', 'Cream', 'Zucchini',
            'Cucumber', 'Winter melon'
        ]

    def _get_model(self, model_path='model.tflite'):
        """ Load the model from a local path """
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def preprocess_image(self, image_path, input_size):
        """Preprocess the input image to feed to the TFLite model"""
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)
        resized_img = tf.image.resize(img, input_size)
        resized_img = resized_img[tf.newaxis, :]
        return resized_img

    def set_input_tensor(self, image):
        """Set the input tensor."""
        tensor_index = self.model.get_input_details()[0]['index']
        input_tensor = self.model.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        """Retur the output tensor at the given index."""
        output_details = self.model.get_output_details()[index]
        tensor = np.squeeze(self.model.get_tensor(output_details['index']))
        return tensor

    def detect_objects(self, image):
        """Returns a list of detection results, each a dictionary of object info."""
        # Feed the input image to the model
        self.set_input_tensor(image)
        self.model.invoke()

        # Get all outputs from the model
        boxes = self.get_output_tensor(1)
        classes = self.get_output_tensor(3)
        scores = self.get_output_tensor(0)
        count = int(self.get_output_tensor(2))

        results = []
        for i in range(count):
            result = {
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            }
            results.append(result)
        return results

    def run_detection(self, image_path, threshold=0.5):
        """Run object detection on the input image and draw the detection results"""
        # Load the input shape required by the model
        _, input_height, input_width, _ = self.model.get_input_details()[0]['shape']

        # Load the input image and preprocess it
        preprocessed_image = self.preprocess_image(image_path, (input_height, input_width))

        # Run object detection on the input image
        results = self.detect_objects(preprocessed_image)
        return results

    def predict(self, image_path, threshold=0.1):
        detection_result_image = self.run_detection(image_path, threshold)
        print(f"Received file for prediction: {image_path}")

        ingredients = []
        filtered_scores = []
        filtered_bboxes = []

        for result in detection_result_image:
            if result['score'] > threshold:
                res_class = self.classes[result['class_id']]
                # do not return redundant ingredients
                if res_class not in ingredients:
                    ingredients.append(res_class)
                    # should still return the highest scores since the original lists are descending ordered
                    filtered_scores.append(float(result['score']))
                    filtered_bboxes.append(result['bounding_box'].tolist())
        return (ingredients, filtered_scores, filtered_bboxes)


if __name__ == '__main__':
    df_test = get_data('path_to_test_data')
    predictor = Predictor()
    y_pred = predictor.predict(df_test)

    print(f"Prediction: {y_pred}")
