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

    with open('metrics/food_testing_set.csv') as csvfile:
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


def download_test_images():
    counter = 0
    ingredients_dict = retrieve_info_from_csv()

    for url in ingredients_dict.values():
        print(f"Download image from {url[1]}")
        image = requests.get(url[1]).content

        with open(f'metrics/images_from_csv/{counter}.jpg', 'wb') as writer:
            writer.write(image)
        counter += 1


def making_prediction():
    prediction_dict = {}
    counter = 0
    img_sort_list = []
    sorted_img_list = []

    for pic in os.listdir("metrics/images_from_csv"):

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
            prediction = predictor.predict(f"metrics/images_from_csv/{img}")
            prediction_dict[f"{counter}"] = prediction
            counter += 1
        except:
            prediction_dict[f"{counter}"] = ["exception"]
            counter += 1

    print(prediction_dict)

    return prediction_dict


def calculating_score():
    ingredients_dict = retrieve_info_from_csv()
    prediction_dict = making_prediction()

    counter = 0
    volumne_counter = 0
    correct_counter = 0
    false_predict_counter = 0
    volumne_pred_counter = 0

    while counter <= 65:

        for items in ingredients_dict[f"{counter}"][0]:
            number_words = items.count(",") + 1

            volumne_counter += number_words

            for pred in prediction_dict[f"{counter}"]:
                pred = pred.lower()
                volumne_pred_counter += 1

                if pred in items:
                    correct_counter += 1
                else:
                    false_predict_counter += 1

        counter += 1

    print(correct_counter, "correct_counter")
    print(volumne_counter, "csv volumne_counter")
    print(volumne_pred_counter, "volumne_pred_counter")
    print(false_predict_counter, "false_predict_counter")

    accuracy = round(correct_counter / volumne_counter, 2)
    print(accuracy, "accuracy")

    perc_false_predictions = round(false_predict_counter / volumne_pred_counter, 2)
    print(perc_false_predictions, "percentage of false predictions")


#retrieve_info_from_csv()
#download_test_images()
making_prediction()
#calculating_score()
