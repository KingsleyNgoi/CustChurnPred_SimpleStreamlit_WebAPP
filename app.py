import os, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import streamlit as st

# os.chdir(r"C:/Users/User/Downloads/streamlit")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
curdir = os.getcwd()
print(f"Current Directory: {curdir}")
print(f"Tensorflow Keras version: {tf.__version__}")

# Load the trained model
model = load_model(r'./model/model_ct.h5')
# Load the encoders and scaler
with open(r'./scaler_encoder/onehot_encoder_ct.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)
with open(r'./scaler_encoder/scaler_ct.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit app
st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', onehot_encoder.named_transformers_['enc_geo'].categories_[0])
gender = st.selectbox('Gender', onehot_encoder.named_transformers_['enc_gen'].categories_[0])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography], 
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]})

# Transform the entire DataFrame
input_data_encoded = onehot_encoder.transform(input_data)  # Correct: Use the entire DataFrame
input_data_scaled = scaler.transform(input_data_encoded)

# Reshape for prediction (if your model requires it)
input_data_scaled = input_data_scaled.reshape(1, -1)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0] #  Or prediction[0, 0] depending on model output

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
