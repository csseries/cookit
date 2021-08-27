import tensorflow as tf
import numpy as np
import cv2

import joblib
from cookit.data import get_data
from PIL import Image

model_path = 'model.tflite'

# Load the labels into a list
classes = ['Cheese', 'Seafood', 'Tomato', 'Baked goods', 'Salad']

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)



class Predictor_lite():
    def __init__(self):
        """
        A basic call for predictions.
        """
        self.model = self._get_model()

    def _get_model(self, model_path="model.tflite"):
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        return hub.load(module_handle).signatures['default']


def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image


def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Retur the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    # Feed the input image to the model
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all outputs from the model
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

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


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
            image_path,
        (input_height, input_width)
        )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

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
    return original_uint8






















# class Predictor():
#     def __init__(self):
#         """
#         A basic call for predictions.
#         """
#         self.model = self._get_model()

#     def _get_model(self, path_to_joblib='model.joblib'):
#         #pipeline = joblib.load(path_to_joblib)
#         #return pipeline
#         module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
#         return hub.load(module_handle).signatures['default']

#     def load_img(self, path):
#         img = tf.io.read_file(path)
#         img = tf.image.decode_jpeg(img, channels=3)
#         return img

#     def predict(self, filename):
#         print(f"Received file for prediction: {filename}")
#         classes, scores = self.run_detector(filename)
#         ingredients_raw = [i.decode('UTF-8') for i in classes]

#         # sort out redundant labels but keep order
#         ingredients = []
#         for i in ingredients_raw:
#             if i not in ingredients:
#                 ingredients.append(i)
#         return ingredients
#         #return ['Cucumber, Carrot, Garlic, Butter, Toast']

#     def run_detector(self, path):
#         img = self.load_img(path)

#         converted_img = tf.image.convert_image_dtype(img,
#                                                      tf.float32)[tf.newaxis,
#                                                                  ...]
#         result = self.model(converted_img)

#         result = {key: value.numpy() for key, value in result.items()}
#         return (result["detection_class_entities"], result["detection_scores"])


# if __name__ == '__main__':
#     df_test = get_data('path_to_test_data')
#     predictor = Predictor()
#     y_pred = predictor.predict(df_test)

#     print(f"Prediction: {y_pred}")
