from typing import List
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import sklearn
import pickle

model = pickle.load(open('DecisionTree.pkl', 'rb'))  # Assuming you want to use DecisionTree model
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    content = """
    <html>
    <head>
        <title>Crop Prediction</title>
    </head>
    <body>
        <h1>Welcome to Crop Prediction</h1>
        <form id="predictionForm" method="post" action="/predict">
            <label for="Nitrogen">Nitrogen:</label>
            <input type="text" id="Nitrogen" name="Nitrogen"><br><br>
            <label for="Phosporus">Phosporus:</label>
            <input type="text" id="Phosporus" name="Phosporus"><br><br>
            <label for="Potassium">Potassium:</label>
            <input type="text" id="Potassium" name="Potassium"><br><br>
            <label for="Temperature">Temperature:</label>
            <input type="text" id="Temperature" name="Temperature"><br><br>
            <label for="Humidity">Humidity:</label>
            <input type="text" id="Humidity" name="Humidity"><br><br>
            <label for="pH">pH:</label>
            <input type="text" id="pH" name="pH"><br><br>
            <label for="Rainfall">Rainfall:</label>
            <input type="text" id="Rainfall" name="Rainfall"><br><br>
            <button type="button" onclick="submitForm()">Predict</button>
        </form>

        <div id="result"></div>

        <script>
            // Define crop_dict as a JavaScript variable
            var crop_dict = {
                1: 'rice',
                2: 'maize',
                3: 'jute',
                4: 'cotton',
                5: 'coconut',
                6: 'papaya',
                7: 'orange',
                8: 'apple',
                9: 'muskmelon',
                10: 'watermelon',
                11: 'grapes',
                12: 'mango',
                13: 'banana',
                14: 'pomegranate',
                15: 'lentil',
                16: 'blackgram',
                17: 'mungbean',
                18: 'mothbeans',
                19: 'pigeonpeas',
                20: 'kidneybeans',
                21: 'chickpea',
                22: 'coffee'
            };

            function submitForm() {
                var form = document.getElementById("predictionForm");
                var formData = new FormData(form);
                fetch("/predict", {
                    method: "POST",
                    body: formData
                }).then(response => response.text())
                .then(data => {
                    document.getElementById("result").innerHTML = data;
                });
            }
        </script>
    </body>
</html>

    """
    return HTMLResponse(content=content)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    feature_list = [float(form["Nitrogen"]), float(form["Phosporus"]), float(form["Potassium"]), 
                    float(form["Temperature"]), float(form["Humidity"]), float(form["pH"]), 
                    float(form["Rainfall"])]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(single_pred)

    """crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}"""
                 
    crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
    }
    if prediction[0] in crop_dict:
        result = f"{crop_dict[prediction[0]]} is the best crop to be cultivated right there"
    else:
        result = "No Crop is predicted"
       
    return HTMLResponse(content=f"<p>{result}</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
