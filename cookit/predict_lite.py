import tensorflow as tf
import numpy as np
import cv2
import os

from cookit.data import get_data
from PIL import Image
from cookit.utils import OIv4_FOOD_CLASSES

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Predictor():
    def __init__(self):
        """
        A basic call for predictions.
        """
        self.model = self._get_model()
    
    def _get_model(self, model_path='model.tflite'):
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        return interpreter

    def image_prep(self, image_path):
        !wget -q -O $image_path #$image_url
        im = Image.open(image_path)
        im.thumbnail((512, 512), Image.ANTIALIAS)
        im.save(image_path, 'PNG')
        return image_path

    def preprocess_image(self, image_path, input_size):
        """Preprocess the input image to feed to the TFLite model"""
        image_path = self.image_prep(image_path)
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)
        original_image = img
        resized_img = tf.image.resize(img, input_size)
        resized_img = resized_img[tf.newaxis, :]
        return resized_img, original_image

    def set_input_tensor(self, interpreter, image):
        """Set the input tensor."""
        tensor_index = interpreter.self.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, interpreter, index):
        """Retur the output tensor at the given index."""
        output_details = interpreter.self.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
        return tensor

    def detect_objects(self, interpreter, image, threshold):
        """Returns a list of detection results, each a dictionary of object info."""
        # Feed the input image to the model
        self.set_input_tensor(interpreter, image)
        interpreter.invoke()

        # Get all outputs from the model
        boxes = self.get_output_tensor(interpreter, 0)
        classes = self.get_output_tensor(interpreter, 1)
        scores = self.get_output_tensor(interpreter, 2)
        count = int(self.get_output_tensor(interpreter, 3))

        results = []
        for i in range(count):
            if scores[i] >= threshold:
              result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
              }
              results.append(result)
        return results

    def run_odt_and_draw_results(self, image_path, interpreter, threshold=0.5):
        """Run object detection on the input image and draw the detection results"""
        # Load the input shape required by the model
        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

        # Load the input image and preprocess it
        preprocessed_image, original_image = self.preprocess_image(
            image_path,
            (input_height, input_width)
        )

        # Run object detection on the input image
        results = self.detect_objects(interpreter, preprocessed_image, threshold=threshold)

        # Plot the detection results on the input image
        original_image_np = original_image.numpy().astype(np.uint8)
        for obj in results:
            # Convert the object bounding box from relative coordinates to absolute
            # coordinates based on the original image resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * original_image_np.shape[1])
            xmax = int(xmax * original_image_np.shape[1])
            ymin = int(ymin * original_image_np.shape[0])
            ymax = int(ymax * original_image_np.shape[0])

            # Find the class index of the current object
            class_id = int(obj['class_id'])

            # Draw the bounding box and label on the image
            color = [int(c) for c in COLORS[class_id]]
            cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
            # Make adjustments to make the label visible for all objects
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
            cv2.putText(original_image_np, label, (xmin, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Return the final image
        original_uint8 = original_image_np.astype(np.uint8)
        return results

    def predict(self, image_path, threshold=0.25):
        interpreter = self._get_model()
        detection_result_image = self.run_odt_and_draw_results(image_path, interpreter, threshold)
        print(f"Received file for prediction: {image_path}")

        # sort out redundant labels but keep order
        ingredients = []
        filtered_scores = []
        filtered_bboxes = []
        res_map = {}

        for i in range(len(classes)):
            res_map[i] = classes[i]

        for item in detection_result_image:
            result = {key: value for key, value in item.items()}
            # add only ingredients which have higher score than threshold
            if float(result['score']) > threshold:
                # add only ingredients which are food-related
                if str(res_map[result['class_id']]) not in ingredients and res_map[result['class_id']] in OIv4_FOOD_CLASSES:
                    res = str(res_map[result['class_id']])
                    ingredients.append(res)
                    # should still return the highest scores since the original lists are descending ordered
                    filtered_scores.append(float(result['score']))
                    filtered_bboxes.append(result['bounding_box'].tolist())
        return (ingredients, filtered_scores, filtered_bboxes)


if __name__ == '__main__':
    df_test = get_data('path_to_test_data')
    predictor = Predictor()
    y_pred = predictor.predict(df_test)

    print(f"Prediction: {y_pred}")
