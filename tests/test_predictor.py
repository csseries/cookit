from cookit.predict import Predictor
import numpy


predictor = Predictor()
test_file = 'tests/images/food.jpg'

def testing_load_img_shape():
    placeholder = predictor.load_img(test_file).shape
    assert predictor.load_img(test_file).shape == [placeholder[0], placeholder[1], 3] # this doesn't make much sense, does it?


def testing_predict():
    res = predictor.predict(test_file)
    assert type(res) == tuple
    assert type(res[0]) == list
    assert type(res[0][0]) == str
    # TODO: test returned scores and bounding boxes too


def testing_run_detector():
    res = predictor.run_detector(test_file)
    assert type(res) == tuple
    assert type(res[0][0]) == bytes
    assert type(res[1][0]) == numpy.float32
