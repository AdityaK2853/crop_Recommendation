import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('DecisionTree.pkl', 'rb'))  

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

st.title('Crop Prediction')

st.write('Enter the following details to predict the best crop to be cultivated:')

nitrogen = st.text_input('Nitrogen')
phosporus = st.text_input('Phosporus')
potassium = st.text_input('Potassium')
temperature = st.text_input('Temperature')
humidity = st.text_input('Humidity')
ph = st.text_input('pH')
rainfall = st.text_input('Rainfall')

if st.button('Predict'):
    feature_list = [float(nitrogen), float(phosporus), float(potassium),
                    float(temperature), float(humidity), float(ph),
                    float(rainfall)]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(single_pred)

    if prediction[0] in crop_dict:
        result = f"{crop_dict[prediction[0]]} is the best crop to be cultivated right there"
    else:
        result = "No Crop is predicted"

    st.write(result)
