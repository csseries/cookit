import os
from uuid import uuid1
from pathlib import Path
import pandas as pd
from termcolor import colored
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from cookit.predict_lite import Predictor


predictor = Predictor()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}


# this expects a POST call like that:
# files = {'image': open('path/to/file/IMG_20150713_195216.jpg', 'rb')}
# payload = {'threshold': 0.1}
# response = requests.post('http://localhost:8000/predict', files=files, data=payload)
# --> make sure to close the opened file again or del the files dictionary in this example case!
@app.post("/predict")
async def predict(image: UploadFile = File(...), threshold: float = Form(0.5)):
    """ Executes prediction based on sent file and threshold
        image: a file in multipart/form-data format
        threshold: a float value (default: 0.25) to filter predicted classes
        probability in form-data

        Returns JSON containing predicted ingredients, their max. scores & bboxes
    """
    filename = str(uuid1()) + '.jpg'
    print(colored(f"API received image {filename} with threshold {threshold}", 'magenta'))

    # create folder if not exists for uploaded images
    Path("./files").mkdir(parents=True, exist_ok=True)
    file_location = f"./files/{filename}"

    # wait until file is completely loaded and store it on disk
    with open(file_location, "wb+") as file_object:
        file_object.write(image.file.read()) # this awaits the read()
        print(colored(f"File stored under {file_location}", 'magenta'))

    # predict ingredients
    y_pred, scores, bboxes = predictor.predict(file_location, threshold)

    # delete uploaded file after prediction
    try:
        os.remove(file_location)
    except OSError as e:  ## if failed, report it back to logs
        print("Error: %s - %s." % (e.filename, e.strerror))

    return {"prediction": y_pred,
            "scores": scores,
            "bboxes": bboxes}
