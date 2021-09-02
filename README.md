![Deploying](https://img.shields.io/github/workflow/status/csseries/cookit/Deploy%20to%20Cloud%20Run/master)


# cookit

cookit is a deep learning model for object-detection of food ingredients on images.
It's running on a webserver, deployed in a Docker container. The model used for predictions is [FasterRCNN+InceptionResNetV2](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1), trained on [OpenImages v4](https://storage.googleapis.com/openimages/web/index.html)

The [frontend repository](https://github.com/csseries/cookit_frontend) is [running here](https://cookit-frontend.herokuapp.com/). A user may upload there an image of certain food ingredients and the deep learning model will detect a list of ingredients which are on the image. The user can than edit and extend ingredients to find recipes which make the most of available ingredients at home.

# Installation

Get the project
```bash
# Either
git clone git@github.com:csseries/cookit.git

# Or
git clone https://github.com/csseries/cookit.git
```

Create virtualenv:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Install requirements:
```bash
pip install -r requirements
```

NOTE: At the time of writing, Mac users with a M1 chip had to apply this command in addition:
```bash
pip uninstall tensorflow
pip install tensorflow-macos tensorflow-metal
```


# Run the backend

Download model (optional):
```bash
make download_model
```


Run the backend
```bash
# Will run backend locally on http://localhost:8080/predict
make run_locally
```

# Performance

To test the performance of the model, run
```bash
make performance_test
```
which will download a number of images, apply the detector and calculate an accuracy value.

# Acknowledgement
This project was made within the scope of a [Le Wagon](https://www.lewagon.com/) Data Science bootcamp, batch #674 in Munich. 🚌

## Made with  ❤️  by
- [Claire-Sophie Sériès](https://github.com/csseries)
- [Justin Bruce Sams](https://github.com/JustinSms)
- [Lilly Kämmerling](https://github.com/lillykml)
- [Judith Reker](https://github.com/judd-r)
- [Michael Weitnauer](https://github.com/kickermeister)
