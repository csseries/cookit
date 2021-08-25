import os
from pathlib import Path
import pandas as pd
from termcolor import colored
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from cookit.predict import Predictor


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
# response = requests.post('http://localhost:8000/predict', files=files)
@app.post("/predict")
async def predict(image: UploadFile = File(...), threshold: float = Form(...)):

    print(colored(f"API received image {image.filename} with threshold {threshold}", 'magenta'))

    # create folder if not exists for uploaded images
    Path("./files").mkdir(parents=True, exist_ok=True)
    file_location = f"./files/{image.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(image.file.read())

    # predict ingredients
    y_pred = predictor.predict(file_location, float(threshold))

    # delete uploaded file after prediction
    try:
        os.remove(file_location)
    except OSError as e:  ## if failed, report it back to logs
        print("Error: %s - %s." % (e.filename, e.strerror))

    return {"prediction": y_pred}
