import os
# import torch

# import albumentations
# import pretrainedmodels

import numpy as np

from flask import Flask
from flask import request
from flask import render_template

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
# from wtfml.data_loaders.image import ClassificationLoader
# from wtfml.engine import Engine

app = Flask(__name__)
UPLOAD_FOLDER = "C:/Users/saipr/Downloads/"

# define custom metric functions
def top_3_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

# register custom metric functions with Keras
custom_objects = {"top_3_accuracy": top_3_accuracy, "top_2_accuracy": top_2_accuracy}
tf.keras.utils.get_custom_objects().update(custom_objects)

model = load_model('model2.h5' )

# class Model 
# predict function

@app.route('/')
def home():
   return render_template('index2.html')


@app.route('/test', methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        # image_file.save('image.png')
        image_file.save(os.path.join(UPLOAD_FOLDER, image_file.filename))

    # image = Image.open('image.png').convert('L')
    image = Image.open(os.path.join(UPLOAD_FOLDER, image_file.filename))

    # resize image to 28x28 pixels
    image = image.resize((224, 224))

    # convert image to numpy array
    img_array = np.array(image)

    # normalize pixel values to range [0, 1]
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    
    # reshape image to 1x28x28
    # img_array = img_array.reshape(1, 224, 224)
    y_pred = model.predict(img_array)
    # print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)

    diseases = {0:'Actinic keratoses and intraepithelial carcinoma', 1: 'Basal cell carcinoma', 2: 'Benign keratosis-like lesions', 3: 'Dermatofibroma', 4: 'Melanoma', 5: 'Melanocytic nevi',6: 'Vascular lesions'}
    
    res = diseases[y_pred[0]]
            
    return render_template("index.html", prediction=res, image_loc=image_file.filename)
    # return render_template("index.html", prediction=0, image_loc=None)



@app.route('/test1', methods=["GET"])
def upload_predict1():                
    return render_template("index.html", prediction="res", image_loc="image_file")

@app.route('/signin', methods=["GET"])
def upload_signin():                
    return render_template("signin2.html")

@app.route('/team', methods=["GET"])
def upload_Team():                
    return render_template("Teams.html")




@app.route('/index2', methods=["GET"])
def upload_index2():                
    return render_template("index2.html")

@app.route('/landing', methods=["GET"])
def upload_landing():                
    return render_template("LandingPage.html")

if __name__ == "__main__":
    app.run(port=12000, debug=True)