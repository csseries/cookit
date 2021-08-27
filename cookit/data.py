import csv
import json
from cookit.utils import OIv4_FOOD_CLASSES, TEST_FOOD_CLASSES

def get_data():
    pass



# Should we perhaps keep non-food-related labels in the test set?
def convert_io_metadta(labelfile_path, baseurl='gs://somewhere', csv_path='tf_training.csv'):
    """ Converts a json file in format fiftyone.types.FiftyOneImageDetectionDataset
        to a format as it is expected by the tflite_model_maker.object_detector

        labelfile_path: path to json file [String]
        baseurl: url to files referenced in the json file [String]
        csv_path: path to output csv file [String]

    """
    with open(labelfile_path) as json_file:
        oi_data = json.load(json_file)

    classes = oi_data['classes']

    with open(csv_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', dialect='excel')
        for uuid, labels in oi_data['labels'].items():
            for label_items in labels:
                label = classes[label_items['label']]
                if label in OIv4_FOOD_CLASSES or label in TEST_FOOD_CLASSES:
                    bb = label_items['bounding_box']
                    line = ['TRAINING', f"{baseurl}/{uuid}.jpg", label, bb[0], bb[1], "", "", bb[2], bb[3], "", ""]
                    writer.writerow(line)

if __name__ == '__main__':
    print("Nothing to do here...")
