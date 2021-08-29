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
    ingredients_dict = {}

    counter = 0

    with open('food_testing_set.csv') as csvfile:
        read = csv.reader(csvfile, delimiter=";")
        for row in read:
            csv_list.append(row)

    for row in csv_list:
        link_list.append(row[0])
        food_list.append(row[1:])

    link_list.pop(0)
    food_list.pop(0)

    for food in food_list:
        ingredients_dict[f"{counter}"] = (food, link_list[counter])

        counter += 1

    return ingredients_dict


def download_test_images(ingredients_dict):
    counter = 0
    #ingredients_dict = retrieve_info_from_csv()

    for url in ingredients_dict.values():
        image = requests.get(url[1]).content

        with open(f'images_from_csv/{counter}.jpg', 'wb') as writer:
            writer.write(image)

    counter += 1


def make_test_prediction():
    pred = predictor.predict("/home/justin/code/JustinSms/cookit/cookit/metrics/images_from_csv/0.jpg")
    print(pred)


def making_prediction():
    prediction_dict = {}
    counter = 0
    img_sort_list = []
    sorted_img_list = []

    for pic in os.listdir("/home/justin/code/JustinSms/cookit/cookit/metrics/images_from_csv"):

        if pic == ".ipynb_checkpoints":
            pass
        else:
            key = int(pic.split(".")[0])
            key_img_tuple = (key, pic)
            img_sort_list.append(key_img_tuple)

    img_sort_list.sort(key=lambda x: x[0])

    for pair in img_sort_list:
        sorted_img_list.append(pair[1])

    for img in sorted_img_list:
        try:
            prediction = predictor.predict(f"/home/justin/code/JustinSms/cookit/cookit/metrics/images_from_csv/{img}")
            prediction_dict[f"{counter}"] = prediction
            counter += 1
        except:
            prediction_dict[f"{counter}"] = ["exception"]
            counter += 1

    print(prediction_dict)

    return prediction_dict


#retrieve_info_from_csv()
#download_test_images(link_list)
#make_test_prediction()
making_prediction()
