FROM python:3.8.6-buster

COPY api /api
COPY cookit /cookit
COPY model.tflite /model.tflite
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install -r requirements.txt

# Download and unpack pre-trained model
RUN make download_model

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
