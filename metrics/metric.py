import csv
from cookit.predict import Predictor
import numpy
import requests
import os


predictor = Predictor()


def retrieve_info_from_csv():
    csv_list = []
    link_list = []
    food_list = []

    with open('food_testing_set.csv') as csvfile:
        read = csv.reader(csvfile, delimiter=";")
        for row in read:
            csv_list.append(row)

    for row in csv_list:
        link_list.append(row[0])
        food_list.append(row[1:])

    link_list.pop(0)
    food_list.pop(0)

    return link_list, food_list

def download_test_images(link_list):
    counter = 0

    for url in link_list:
        image = requests.get(url).content

        with open(f'images_from_csv/{counter}.jpg', 'wb') as writer:
            writer.write(image)

    counter += 1


def making_prediction():
    prediction_dict = {}
    counter = 0

    for pic in os.listdir("images_from_csv"):
        prediction = predictor.predict(pic)
        prediction_dict[f"{counter}"] = prediction

    counter += 1


making_prediction()
