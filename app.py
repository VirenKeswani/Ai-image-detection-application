from flask import Flask, render_template, request
import tensorflow as tf
import os
import cv2
from keras_preprocessing import image
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/semantic', methods=['POST'])
def predict_model_semantic():
    model = tf.keras.models.load_model('./model/res.h5')

    if 'picture' not in request.files:
        return "No file part"
    
    file = request.files['picture']

    if file.filename == '':
        return 'No selected file'
    
    if file:
        file.save('./test.jpg')
        img = cv2.imread('./test.jpg')
        img = cv2.resize(img, (128, 128))  # Resize the image
        x = image.img_to_array(img)
        x /= 255.0
        x = np.expand_dims(x, axis=0)

        # Get the model's prediction
        classes = model.predict(x,verbose=False)
        a = classes[0]
        os.remove('./test.jpg')

        print("accuracy = ",a)
        if(a>=0.5):
            return render_template('index.html', result_semantic='real')
        else:
            return render_template('index.html', result_semantic='fake')

@app.route('/nsemantic', methods=['POST'])
def predict_model_nsemantic():
    model = tf.keras.models.load_model('./model/res.h5')

    if 'picture' not in request.files:
        return "No file part"
    
    file = request.files['picture']

    if file.filename == '':
        return 'No selected file'
    
    if file:
        file.save('./test.jpg')
        img = cv2.imread('./test.jpg')
        img = cv2.resize(img, (128, 128))  # Resize the image
        x = image.img_to_array(img)
        x /= 255.0
        x = np.expand_dims(x, axis=0)
        os.remove('./test.jpg')
        # Get the model's prediction
        classes = model.predict(x,verbose=False)
        a = classes[0]
        print("accuracy = ",a)
        if(a>=0.5):
            return render_template('index.html', result_nsemantic='real')
        else:
            return render_template('index.html', result_nsemantic='fake')
        

if __name__ == '__main__':
    app.run(debug=True, port=8800)
    