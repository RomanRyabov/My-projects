from flask import Flask, request, jsonify
import pickle
import numpy as np

with open ("xgbclassifier.pkl", "rb") as pkl_file:
    model = pickle.load(pkl_file)
    
app = Flask(__name__)
@app.route("/predict", methods=["POST"])

def predict():
    features = np.array(request.json).reshape(1, 174)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run("localhost", 5000)