import csv
import json
import math
from cookit.utils import OIv4_FOOD_CLASSES, OIv4_INGREDIENTS_ONLY, TEST_FOOD_CLASSES

def get_data():
    pass



# Should we perhaps keep non-food-related labels in the test set?
def convert_io_metadta(labelfile_path, baseurl='gs://somewhere', csv_path='tf_training.csv',
                       test_split=0.2, val_split=0.1):
    """ Converts a json file in format fiftyone.types.FiftyOneImageDetectionDataset
        to a format as it is expected by the tflite_model_maker.object_detector

        labelfile_path: path to json file [String]
        baseurl: url to files referenced in the json file [String]
        csv_path: path to output csv file [String]
        test_split: share of images for test [float]
        val_split: share of images for validation [float]

        Returns dict containing basic dataset information
    """
    with open(labelfile_path) as json_file:
        oi_data = json.load(json_file)

    classes = oi_data['classes']
    food_classes = []

    total_images = len(oi_data['labels'])
    val_count = math.floor(total_images* val_split)
    test_count = math.floor(total_images * test_split)

    bbox_count, uuid_count = 0, 0
    test_bboxes, val_bboxes, train_bboxes = 0, 0, 0

    with open(csv_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', dialect='excel')
        line = ['set', 'path', 'label', 'x_min', 'y_min', 'x_max', 'y_min', 'x_max', 'y_max', 'x_min', 'y_max']
        writer.writerow(line)
        for uuid, labels in oi_data['labels'].items():
            uuid_count += 1
            for label_items in labels:
                label = classes[label_items['label']]
                if label in OIv4_INGREDIENTS_ONLY or label in TEST_FOOD_CLASSES:
                    food_classes.append(label)
                    bbox_count += 1
                    bb = label_items['bounding_box']
                    if uuid_count <= val_count:
                        split = "TEST"
                        val_bboxes += 1
                    elif uuid_count >= val_count and uuid_count < int(test_count + val_count):
                        split = "VALIDATION"
                        test_bboxes += 1
                    elif uuid_count >= int(test_count + val_count):
                        split = "TRAINING"
                        train_bboxes += 1
                    line = [split, f"{baseurl}/{uuid}.jpg", label, bb[0], bb[1], "", "", bb[2], bb[3], "", ""]
                    writer.writerow(line)

    unique_food_classes = list(sorted(set(food_classes)))
    print(f"Found {len(unique_food_classes)} food-related classes of total {len(classes)} classes.")
    print(f"Found {bbox_count} bounding boxes. Train/Val/Test split: {train_bboxes} / {val_bboxes} / {test_bboxes}")
    print(f"Total number of images in dataset: {total_images}")

    return {
        'food_classes': unique_food_classes,
        'bbox_count': bbox_count,
        'train_bbox_count': train_bboxes,
        'val_bbox_count': val_bboxes,
        'test_bbox_count': test_bboxes,
        'images_count': total_images
    }

if __name__ == '__main__':
    print("Nothing to do here...")
