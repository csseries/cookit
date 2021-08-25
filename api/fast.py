from pathlib import Path
import pandas as pd
from fastapi import FastAPI, File, UploadFile
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
async def predict(image: UploadFile = File(...)):
    Path("/files").mkdir(parents=True, exist_ok=True)
    file_location = f"/files/{image.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(image.file.read())
    y_pred = predictor.predict(file_location)
    return {"prediction": y_pred}
    #return {"prediction": "Cucumber"}
