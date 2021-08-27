from cookit.predict import Predictor
import numpy


predictor = Predictor()


def testing_load_img_shape():
    placeholder = predictor.load_img("raw_data/test_pic.jpg").shape
    assert predictor.load_img("raw_data/test_pic.jpg").shape == [placeholder[0], placeholder[1], 3]


def testing_predict():
    assert type(predictor.predict("raw_data/test_pic.jpg")) == list
    assert type(predictor.predict("raw_data/test_pic.jpg")[0]) == str


def testing_run_detector():
    assert type(predictor.run_detector("raw_data/test_pic.jpg")) == tuple
    assert type(predictor.run_detector("raw_data/test_pic.jpg")[0][0]) == bytes
    assert type(predictor.run_detector("raw_data/test_pic.jpg")[1][0]) == numpy.float32
