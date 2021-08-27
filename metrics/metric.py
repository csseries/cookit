import csv
from cookit.predict import Predictor
import numpy


csv_list = []
link_list = []
food_list = []

predictor = Predictor()


with open('food_testing_set.csv') as csvfile:
    read = csv.reader(csvfile, delimiter=";")
    for row in read:
        csv_list.append(row)

for row in csv_list:
    link_list.append(row[0])
    food_list.append(row[1:])

link_list.pop(0)
food_list.pop(0)


def calculating_accuracy():

    prediction = predictor.predict()
