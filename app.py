# Import the Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import streamlit as st

# Load the model 
model = tf.keras.models.load_model('model.h5')

# Load the encoder 
with open('label_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app

st.title("Student Performance Index Predictor")

# user input
hours_studies = st.number_input('Hours Studied')
previous_scores = st.number_input('Previous Scores')
extra_activities = st.selectbox('Extracurricular Activities', encoder.classes_)
sleep = st.number_input('Sleep Hours')
papers = st.slider('Sample Paper Practiced', 0, 15)

# input Data
input_data = {
    'Hours Studied': hours_studies,
    'Previous Scores': previous_scores,
    'Extracurricular Activities': extra_activities,
    'Sleep Hours': sleep,
    'Sample Question Papers Practiced': papers
}
input_df = pd.DataFrame([input_data])

# encode the data
input_df['Extracurricular Activities'] = encoder.transform(input_df['Extracurricular Activities'])

# scale the data
input_df = scaler.transform(input_df)

# Predictions
predictions = model.predict(input_df)

st.write(f"Predicted Performance Index of the student is {predictions[0][0]:.2f}")