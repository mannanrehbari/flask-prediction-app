# Predictions performed by this module

#dependencies
import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array

from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    """
    This function loads the already-built keras model
    """
    global model
    model = load_model('model.h5')
    print("Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model ... ")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    """
    whenever something is posted from /predict,
    this function will process the info posted through POST http method
    message: json from POST method
    encoded: key is 'image', value is base64encoded image sent from client
    decoded: as it says
    image: decoded is bytes in a file, not an actual image,
            image.open converts those bytes into PIL file

    """

    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224,224))

    prediction = model.predict(processed_image).tolist()
    response = {
        'prediction':{
            'dog' : prediction[0][0],
            'cat' : prediction[0][1]
        }
    }
    return jsonify(response)
