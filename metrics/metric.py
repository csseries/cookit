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
    # function creates the ingredients_dict for later comparison to the predictions
    csv_list = []
    link_list = []
    food_list = []
    ingredients_dict = {}

    counter = 0
    offset = 1
    if nr_images == -1:
        offset = 0

    with open(f"{rel_path}/food_testing_set.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        csv_list = list(reader)[0:nr_images+offset]

    for row in csv_list:
        link_list.append(row[0])
        food_list.append([ingr.strip() for ingr in row[1].split(',')])

    link_list.pop(0)
    food_list.pop(0)

    for food in food_list:
        ingredients_dict[f"{counter}"] = (food, link_list[counter])

        counter += 1

    return ingredients_dict


def download_test_images(ingredients_dict):
    print(f"Download {len(ingredients_dict)} images")
    counter = 0
    # create local folder for downloads if not exists
    Path(image_path).mkdir(parents=True, exist_ok=True)

    for url in ingredients_dict.values():
        image = requests.get(url[1]).content

        with open(f'{image_path}/{counter}.jpg', 'wb') as writer:
            writer.write(image)
        counter += 1


def making_prediction(nr_images=-1):
    prediction_dict = {}
    counter = 0
    offset = 1
    if nr_images == -1:
        offset = 0

    img_sort_list = []
    sorted_img_list = []

    for pic in os.listdir(image_path)[0:nr_images+offset]:

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
            prediction, scores, bboxes = predictor.predict(f"{image_path}/{img}")
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

    total_amount_ingredients_counter = 0 #len([item[0] for item in ingredients_dict.values()])
    total_amount_predictions_counter = 0
    correct_predict_counter = 0
    correct_oi_predict_counter = 0
    false_predict_counter = 0
    false_oi_predict_counter = 0

    # specific to oi_classes
    oi_ingredients_list_volumne_counter = 0

    for key, value in ingredients_dict.items():
        ingredients = value[0]
        number_words = len(ingredients)
        total_amount_ingredients_counter += number_words
        predictions = prediction_dict[key]

        for pred in predictions:
            # iterate through every prediction and check if it is in ingredients
            pred = pred.lower()
            total_amount_predictions_counter += 1

            if pred in ingredients:
                correct_predict_counter += 1
            else:
                false_predict_counter += 1



        oi_ingredients_list = []

        print(oi_ingredients_list, "oi_ingrdients_list --> empty")

        for ingr in ingredients:
            if ingr in oi_classes:
                oi_ingredients_list.append(ingr)

        print(oi_ingredients_list, "full with ingrdients from oi_classes")
        oi_ingredients_list_volumne_counter += len(oi_ingredients_list)

        for pred in predictions:
            pred = pred.lower()
            print(pred, "predictions")
            if pred in oi_ingredients_list:
                print(pred, "pred correct")
                correct_oi_predict_counter += 1
            else:
                print(pred, "pred false")
                false_oi_predict_counter += 1

        oi_ingredients_list.clear()
        print(oi_ingredients_list, "oi_ingredirents_list --> clear")






    print("##############################################")
    print("Amount of correct predictions: ", correct_predict_counter)
    print("Amount of false predictions: ", false_predict_counter)
    print("Total amount of ingredients in csv file: ", total_amount_ingredients_counter)
    print("Total amount of predictions made: ", total_amount_predictions_counter)
    print("Amount of correct predictions found in algorithm classes: ",correct_oi_predict_counter)
    print("Amount of false predictions found in algorithm classes: ",false_oi_predict_counter)

    accuracy = round(correct_predict_counter / total_amount_ingredients_counter, 2)
    print("Accuracy: ",accuracy)

    perc_false_predictions = round(false_predict_counter / total_amount_predictions_counter, 2)
    print("Percentage of false predictions: ",perc_false_predictions*100)
    print("                                              ")
    print("---------------Oi Classes Metrics-------------")
    print("Amount of correct predictions of oi classes: ",correct_oi_predict_counter)
    print("Oi classes accuracy: ",round(correct_oi_predict_counter / oi_ingredients_list_volumne_counter,2))
    print("Oi classes percentage of wrong predictions: ",round(false_oi_predict_counter / total_amount_predictions_counter,2))
    print("##############################################")

    return accuracy, perc_false_predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest='nr_images', type=int, default=-1,
                        help="Number of images to be downloaded and tested")
    args = parser.parse_args()

    print(f"Calcuclate score based on {'all' if args.nr_images == -1 else args.nr_images} images")
    calculating_score(nr_images=args.nr_images)

    # delete downloaded files after prediction
    try:
        shutil.rmtree(image_path)
    except OSError as e:  ## if failed, report it back to logs
        print("Error: %s - %s." % (e.filename, e.strerror))
