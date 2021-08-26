from cookit.predict import Predictor
import numpy


predictor = Predictor()


def testing_load_img_shape():
    placeholder = predictor.load_img("raw_data/test_pic.jpg").shape
    assert predictor.load_img("raw_data/test_pic.jpg").shape == [placeholder[0], placeholder[1], 3]


def testing_predict_return_list():
    assert type(predictor.predict("raw_data/test_pic.jpg")) == list


def testing_predict_return_str():
    assert type(predictor.predict("raw_data/test_pic.jpg")[0]) == str


def testing_run_detector_return_tuple():
    assert type(predictor.run_detector("raw_data/test_pic.jpg")) == tuple


def testing_run_detector_return_bytes():
    assert type(predictor.run_detector("raw_data/test_pic.jpg")[0][0]) == bytes


def testing_run_detector_return_numpyfloat32():
    assert type(predictor.run_detector("raw_data/test_pic.jpg")[1][0]) == numpy.float32


testing_load_img_shape()
testing_predict_return_list()
testing_predict_return_str()
testing_run_detector_return_tuple()
testing_run_detector_return_bytes()
testing_run_detector_return_numpyfloat32()
