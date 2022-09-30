from flask import Flask, jsonify, request
from keras_preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
import numpy

# creating a Flask app 

app = Flask(__name__)

@app.route('/sentiment', methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return jsonify({"data":"Hello World"})

    elif request.method == "POST":

        text = request.json.get("text")
    
        # loding tokenizer 
        with open("token.pkl","rb") as file:
            tokens = pickle.load(file)
        
        # print(type([text]))

        twt = tokens.texts_to_sequences([text])
        twt = pad_sequences(twt, maxlen=147)

        # loding model 

        model = tf.keras.models.load_model("model.h5")

        result = model.predict(twt)

        print(result)

        if numpy.argmax(result[0]):
            return jsonify({"result":"positive"})
        else:
            return jsonify({"result":"negative"})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
