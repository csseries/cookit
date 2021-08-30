import csv
from pathlib import Path
import requests
import os
import shutil
import argparse
from cookit.utils import OIv4_FOOD_CLASSES
from cookit.predict import Predictor

oi_classes = [ingr.lower() for ingr in OIv4_FOOD_CLASSES]

rel_path = os.path.dirname(__file__)
test_folder = "images_from_csv"
image_path = os.path.join(rel_path, test_folder)

# instanciating here causes very long startup time
predictor = Predictor()


def retrieve_info_from_csv(nr_images=-1):
    csv_list = []
    link_list = []
    food_list = []
    ingredients_dict = {}

    counter = 0

    with open(f"{rel_path}/food_testing_set.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        csv_list = list(reader)[0:nr_images+1]

    for row in csv_list:
        link_list.append(row[0])
        food_list.append([ingr.strip() for ingr in row[1].split(',')])

    link_list.pop(0)
    food_list.pop(0)

    for food in food_list:
        ingredients_dict[f"{counter}"] = (food, link_list[counter])

        counter += 1

    print(ingredients_dict)
    return ingredients_dict


def download_test_images(ingredients_dict):
    print(f"Download {len(ingredients_dict)} images")
    counter = 0
    # create local folder for downloads if not exists
    Path(image_path).mkdir(parents=True, exist_ok=True)

    for url in ingredients_dict.values():
        print(f"Download image from {url[1]}")
        image = requests.get(url[1]).content

        with open(f'{image_path}/{counter}.jpg', 'wb') as writer:
            writer.write(image)
        counter += 1


def making_prediction(nr_images=-1):
    prediction_dict = {}
    counter = 0
    img_sort_list = []
    sorted_img_list = []

    for pic in os.listdir(image_path)[0:nr_images]:

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
            print(f"Predict image {img}")
            prediction, scores, bboxes = predictor.predict(f"{image_path}/{img}")
            print(f"Found ingredients in {img}: {prediction}")
            prediction_dict[f"{counter}"] = prediction
            counter += 1
        except:
            prediction_dict[f"{counter}"] = ["exception"]
            counter += 1

    return prediction_dict


def calculating_score(nr_images=-1):
    ingredients_dict = retrieve_info_from_csv(nr_images)
    download_test_images(ingredients_dict) # passing nr_images is done for safety reasons only
    prediction_dict = making_prediction(nr_images)  # passing nr_images is done for safety reasons only

    volumne_counter = 0 #len([item[0] for item in ingredients_dict.values()])
    correct_counter = 0
    correct_counter_oi = 0
    false_predict_counter = 0
    false_predict_counter_oi = 0
    volumne_pred_counter = 0


    for key, value in ingredients_dict.items():
        ingredients = value[0]
        number_words = len(ingredients)
        volumne_counter += number_words
        predictions = prediction_dict[key]

        for pred in predictions:
            pred = pred.lower()
            volumne_pred_counter += 1
            print(f"Compare {pred} against {ingredients}")
            if pred in ingredients:
                correct_counter += 1
            else:
                false_predict_counter += 1

            if pred in oi_classes:
                correct_counter_oi += 1
            else:
                false_predict_counter_oi += 1

    print("##############################################")
    print(correct_counter, "correct_counter")
    print(volumne_counter, "csv volumne_counter")
    print(volumne_pred_counter, "volumne_pred_counter")
    print(false_predict_counter, "false_predict_counter")

    accuracy = round(correct_counter / volumne_counter, 2)
    print(accuracy, "accuracy")

    perc_false_predictions = round(false_predict_counter / volumne_pred_counter, 2)
    print(perc_false_predictions, "percentage of false predictions")
    print("##############################################")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest='nr_images', type=int, default=-1,
                        help="Number of images to be downloaded and tested")
    args = parser.parse_args()

    print(f"Calcuclate score based on {args.nr_images} images")
    calculating_score(nr_images=args.nr_images)

    # delete downloaded files after prediction
    try:
        shutil.rmtree(image_path)
    except OSError as e:  ## if failed, report it back to logs
        print("Error: %s - %s." % (e.filename, e.strerror))
